"""
MonuScript - Confidence and Rejection Mechanism
Prevents false acceptances when non-monument, unsupported, ambiguous,
or out-of-distribution images are submitted to the SVM classifier.
"""

import numpy as np

# Tuned default thresholds based on validation benchmarks
DEFAULT_ACCEPT_THRESHOLD = 0.40      # Minimum calibrated recognition score for acceptance
DEFAULT_MARGIN_THRESHOLD = 0.12      # Minimum difference between top-1 and top-2 candidate
DEFAULT_UNCERTAIN_THRESHOLD = 0.28   # Below this is outright rejection; between this and accept is uncertain

REJECTION_MESSAGE = (
    "Oh my! Haven't seen this one before! 🧐\n\n"
    "Sorry, I couldn't confidently identify a supported monument in this image. "
    "Try uploading another photograph."
)

UNCERTAIN_MESSAGE_TEMPLATE = (
    "This photograph has some structural resemblance to {candidate}, "
    "but the recognition score ({score:.0%}) is below our acceptance threshold. "
    "Try uploading a clearer, unobstructed angle."
)

class RejectionJudge:
    def __init__(
        self,
        accept_threshold=DEFAULT_ACCEPT_THRESHOLD,
        margin_threshold=DEFAULT_MARGIN_THRESHOLD,
        uncertain_threshold=DEFAULT_UNCERTAIN_THRESHOLD,
        class_centroids=None
    ):
        self.accept_threshold = accept_threshold
        self.margin_threshold = margin_threshold
        self.uncertain_threshold = uncertain_threshold
        self.class_centroids = class_centroids  # dict of class_id -> mean feature vector in scaled space

    def evaluate(self, probabilities, class_names, scaled_features=None):
        """
        Evaluates model output for rejection or acceptance.
        Parameters:
            probabilities: 1D array of class probabilities / calibrated scores summing to ~1
            class_names: list of class names in corresponding order
            scaled_features: optional 1D array of features in scaled space
        Returns dict:
            {
                "accepted": bool,
                "is_uncertain": bool,
                "top_class": str,
                "top_score": float,
                "runner_up_class": str or None,
                "runner_up_score": float or None,
                "score_margin": float,
                "rejection_reason": str or None,
                "message": str
            }
        """
        probs = np.array(probabilities, dtype=np.float32)
        sorted_indices = np.argsort(probs)[::-1]

        top_idx = sorted_indices[0]
        top_class = class_names[top_idx]
        top_score = float(probs[top_idx])

        if len(sorted_indices) > 1:
            runner_idx = sorted_indices[1]
            runner_class = class_names[runner_idx]
            runner_score = float(probs[runner_idx])
            margin = top_score - runner_score
        else:
            runner_class = None
            runner_score = 0.0
            margin = top_score

        # Out-of-distribution distance check if centroids are provided
        is_ood = False
        if self.class_centroids is not None and scaled_features is not None:
            centroid = self.class_centroids.get(top_class)
            if centroid is not None:
                dist = np.linalg.norm(scaled_features - centroid)
                # If distance exceeds 3 standard deviations in feature space (approx dist > 20.0 for 139 dims)
                if dist > 22.0:
                    is_ood = True

        # Decision rules
        if is_ood or top_score < self.uncertain_threshold:
            return {
                "accepted": False,
                "is_uncertain": False,
                "top_class": top_class,
                "top_score": top_score,
                "runner_up_class": runner_class,
                "runner_up_score": runner_score,
                "score_margin": margin,
                "rejection_reason": "out_of_distribution" if is_ood else "low_confidence",
                "message": REJECTION_MESSAGE
            }

        if top_score < self.accept_threshold or margin < self.margin_threshold:
            # Uncertain / Ambiguous
            return {
                "accepted": False,
                "is_uncertain": True,
                "top_class": top_class,
                "top_score": top_score,
                "runner_up_class": runner_class,
                "runner_up_score": runner_score,
                "score_margin": margin,
                "rejection_reason": "ambiguous_margin" if margin < self.margin_threshold else "below_accept_threshold",
                "message": UNCERTAIN_MESSAGE_TEMPLATE.format(candidate=top_class.replace("_", " ").title(), score=top_score)
            }

        # Accepted
        return {
            "accepted": True,
            "is_uncertain": False,
            "top_class": top_class,
            "top_score": top_score,
            "runner_up_class": runner_class,
            "runner_up_score": runner_score,
            "score_margin": margin,
            "rejection_reason": None,
            "message": f"Successfully identified {top_class.replace('_', ' ').title()} with recognition score {top_score:.1%}."
        }
