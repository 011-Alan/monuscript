import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, X, Sparkles, Loader2, AlertCircle } from 'lucide-react';

interface ImageUploaderProps {
  onUpload: (file: File) => Promise<void>;
  isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onUpload, isLoading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File) => {
    setErrorMessage(null);
    // Allowed extensions
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp|tiff?)$/i)) {
      setErrorMessage("That file format isn't supported yet. Please upload a valid image (JPEG, PNG, WEBP, BMP).");
      return;
    }
    // Size limit 16MB
    if (file.size > 16 * 1024 * 1024) {
      setErrorMessage('The image file is too large (maximum 16 MB).');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || isLoading) return;
    try {
      await onUpload(selectedFile);
      handleClear();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to analyze image.');
    }
  };

  return (
    <div className="border-t border-[#E5DFD7] bg-white/80 backdrop-blur-sm p-4">
      <div className="max-w-3xl mx-auto">
        {errorMessage && (
          <div className="mb-3 p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
            <span>{errorMessage}</span>
          </div>
        )}

        {previewUrl ? (
          /* Preview and Submit State */
          <div className="flex flex-col sm:flex-row items-center justify-between p-3 rounded-xl border border-amber-300 bg-amber-50/50 gap-3">
            <div className="flex items-center space-x-3 w-full sm:w-auto">
              <div className="relative w-16 h-16 rounded-lg overflow-hidden bg-stone-200 shrink-0 border border-stone-300">
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-stone-800 truncate max-w-[200px] sm:max-w-xs">
                  {selectedFile?.name}
                </p>
                <p className="text-[11px] text-stone-500">
                  {((selectedFile?.size || 0) / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2 w-full sm:w-auto justify-end">
              <button
                type="button"
                onClick={handleClear}
                disabled={isLoading}
                className="px-3 py-2 rounded-lg text-xs font-medium text-stone-600 hover:bg-stone-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isLoading}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-medium bg-[#C85A32] hover:bg-[#b04d28] text-white shadow-xs transition-all active:scale-[0.99] disabled:opacity-70"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Examining architectural features...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Analyze Image</span>
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* Drag & Drop Input Area */
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              relative flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl cursor-pointer transition-colors
              ${isDragOver 
                ? 'border-amber-500 bg-amber-50/50' 
                : 'border-[#D5CABE] hover:border-amber-500 bg-sand-50/50 hover:bg-amber-50/20'}
            `}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
              accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
              className="hidden"
            />
            <div className="flex items-center space-x-3 text-stone-600">
              <div className="p-2 rounded-lg bg-white border border-[#E5DFD7] text-[#C85A32]">
                <Upload className="w-5 h-5" />
              </div>
              <div className="text-left">
                <p className="text-xs font-semibold text-stone-800">
                  Select or drop a monument photograph
                </p>
                <p className="text-[11px] text-stone-500">
                  JPEG, PNG, WEBP, or BMP • Evaluated by Image Processing Syllabus pipeline
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
