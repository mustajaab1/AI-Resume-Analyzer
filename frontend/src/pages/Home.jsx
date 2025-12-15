import React, { useRef, useState } from 'react';
import { Upload, FileText, CheckCircle2, Circle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const Home = () => {
    const navigate = useNavigate();
    const fileInputRef = useRef(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    };

    const handleFileClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        // In a real app, we'd upload the file here.
        // For now, we simulate and navigate to analyze
        console.log("File selected:", file.name);
        navigate('/analyze', { state: { fileName: file.name } });
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="max-w-3xl mx-auto space-y-8"
            >
                <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
                    Is your resume <br />
                    <span className="text-gradient">AI-Ready?</span>
                </h1>

                <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto">
                    Unlock your career potential with our advanced resume analyzer.
                    Get instant feedback on keywords, formatting, and ATS compatibility.
                </p>

                <div className="pt-8">
                    <div
                        onClick={handleFileClick}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`
              relative group cursor-pointer
              w-full max-w-2xl mx-auto
              h-64 rounded-3xl
              border-2 border-dashed transition-all duration-300
              flex flex-col items-center justify-center gap-4
              glass-card
              ${isDragging ? 'border-primary bg-primary/5' : 'border-white/10 hover:border-primary/50 hover:bg-dark-800/80'}
            `}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept=".pdf,.docx,.doc"
                            className="hidden"
                        />

                        <div className="p-4 rounded-full bg-dark-900/50 border border-white/5 shadow-inner group-hover:scale-110 transition-transform duration-300">
                            <Upload className="w-10 h-10 text-primary" />
                        </div>

                        <div className="space-y-2">
                            <p className="text-xl font-medium text-white">
                                Drag & drop or click to upload resume
                            </p>
                            <p className="text-sm text-gray-500 uppercase tracking-wider">
                                (PDF / DOCX)
                            </p>
                        </div>
                    </div>
                </div>

                <div className="pt-12 flex flex-wrap justify-center gap-6 md:gap-12">
                    {[
                        { label: 'ATS Compatibility Check', active: true },
                        { label: 'Keyword Extraction', active: true },
                        { label: 'Trend Analysis', active: true }
                    ].map((step, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm md:text-base font-medium text-gray-300">
                            {step.active ? (
                                <CheckCircle2 className="w-5 h-5 text-green-500" />
                            ) : (
                                <Circle className="w-5 h-5 text-gray-600" />
                            )}
                            {step.label}
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
};

export default Home;
