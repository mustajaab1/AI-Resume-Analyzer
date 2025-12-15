import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, FileText, AlertTriangle, Check, X, Download } from 'lucide-react';

const Analyzer = () => {
    const [analysisData, setAnalysisData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    const location = useLocation();
    const file = location.state?.file;
    const analysisType = location.state?.type || 'deep';

    useEffect(() => {
        if (!file) {
            setError("No file provided");
            setLoading(false);
            return;
        }

        const analyzeFile = async () => {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('type', analysisType);

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`Analysis failed: ${response.statusText}`);
                }

                const data = await response.json();
                setAnalysisData(data);
            } catch (err) {
                console.error("Analysis Error:", err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        analyzeFile();
    }, [file]);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[70vh]">
                <div className="relative">
                    {/* Glowing Orb Animation */}
                    <div className="w-32 h-32 rounded-full bg-gradient-to-tr from-primary to-accent blur-md animate-spin-slow opacity-80" />
                    <div className="absolute inset-0 w-32 h-32 rounded-full border-4 border-white/20 border-t-primary animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Brain className="w-12 h-12 text-white animate-pulse" />
                    </div>
                </div>
                <h2 className="mt-8 text-2xl font-semibold text-white">Processing Document...</h2>
                <p className="text-gray-400 mt-2">Running NLP extraction and ATS compliance checks.</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[70vh] text-center">
                <div className="p-4 rounded-full bg-red-500/10 mb-4">
                    <X className="w-12 h-12 text-red-500" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Analysis Failed</h2>
                <p className="text-gray-400 max-w-md">{error}</p>
                <button
                    onClick={() => window.history.back()}
                    className="mt-6 px-6 py-2 bg-dark-800 hover:bg-dark-700 rounded-lg text-white transition-colors"
                >
                    Try Again
                </button>
            </div>
        );
    }

    const {
        score = 0,
        job_title = "General",
        summary = "No summary available.",
        detected_keywords = [],
        missing_skills = [],
        recommended_keywords = [],
        learning_resources = {},
        detailed_feedback = [],
        validation = {}
    } = analysisData || {};

    // Use missing_skills if available, otherwise recommended_keywords
    const missing = missing_skills.length > 0 ? missing_skills : recommended_keywords;

    return (
        <div className="space-y-6 animate-fade-in">
            <header className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-2xl font-bold">Analysis Report</h1>
                    <p className="text-gray-400 text-sm">Target Role: <span className="text-primary font-medium">{job_title}</span></p>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-dark-800 rounded-lg hover:bg-dark-700 border border-white/10 transition-colors">
                    <Download size={16} /> Export PDF
                </button>
            </header>

            {validation?.is_valid === false && (
                <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-200">
                    <div className="flex items-center gap-2 font-semibold mb-2">
                        <AlertTriangle className="w-5 h-5" />
                        <span>Resume Validation Warning</span>
                    </div>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                        {validation.issues?.map((issue, idx) => <li key={idx}>{issue}</li>)}
                    </ul>
                </div>
            )}

            {/* Top Row: Score & Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Score Card */}
                <div className="glass-card p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                    <div className="relative w-40 h-40 flex items-center justify-center">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                            <circle
                                cx="50" cy="50" r="45"
                                fill="none"
                                stroke="#1c1c2e"
                                strokeWidth="10"
                            />
                            <circle
                                cx="50" cy="50" r="45"
                                fill="none"
                                stroke={score > 70 ? "#22c55e" : score > 40 ? "#eab308" : "#ef4444"}
                                strokeWidth="10"
                                strokeDasharray="283"
                                strokeDashoffset={isNaN(score) ? 283 : 283 - (283 * score) / 100}
                                strokeLinecap="round"
                                className={`drop-shadow-[0_0_10px_rgba(${score > 70 ? "34,197,94" : score > 40 ? "234,179,8" : "239,68,68"},0.5)]`}
                            />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-4xl font-bold">{score}</span>
                            <span className="text-xs text-gray-400 uppercase tracking-widest mt-1">Match Score</span>
                        </div>
                    </div>
                </div>

                {/* Baseline Summary */}
                <div className="glass-card p-6 rounded-2xl lg:col-span-2 space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                        <FileText className="text-primary" size={20} />
                        <h3 className="text-lg font-semibold">Executive Summary</h3>
                    </div>
                    <p className="text-gray-300 leading-relaxed">
                        {summary}
                    </p>
                </div>
            </div>

            {/* Keyword Analysis Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Detected Keywords */}
                <div className="glass-card p-6 rounded-2xl">
                    <div className="flex items-center gap-2 mb-4">
                        <Check className="text-green-400" size={20} />
                        <h3 className="text-lg font-semibold text-green-400">Your Skills</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {Array.isArray(detected_keywords) && detected_keywords.map(tag => (
                            <span key={tag} className="px-3 py-1 rounded-full text-sm bg-green-500/10 text-green-300 border border-green-500/20">
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Missing Skills / Skill Gap */}
                <div className="glass-card p-6 rounded-2xl">
                    <div className="flex items-center gap-2 mb-4">
                        <AlertTriangle className="text-orange-400" size={20} />
                        <h3 className="text-lg font-semibold text-orange-400">Missing Skills (Skill Gap)</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {Array.isArray(missing) && missing.map(tag => (
                            <span key={tag} className="px-3 py-1 rounded-full text-sm bg-orange-500/10 text-orange-300 border border-orange-500/20 dashed-border">
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>
            </div>

            {/* Detailed Feedback & Learning Resources Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Detailed Feedback */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-blue-400">Detailed Resume Feedback</h3>
                    <ul className="space-y-4">
                        {Array.isArray(detailed_feedback) && detailed_feedback.length > 0 ? (
                            detailed_feedback.map((item, idx) => (
                                <li key={idx} className="flex items-start gap-3 text-sm text-gray-300 bg-dark-800/50 p-3 rounded-lg">
                                    <div className="mt-1">👉</div>
                                    <span>{item}</span>
                                </li>
                            ))
                        ) : (
                            <li className="text-gray-500 italic">No specific feedback generated.</li>
                        )}
                    </ul>
                </div>

                {/* Learning Resources / Career Advisor */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-purple-400">Recommended Learning Paths</h3>
                    <div className="space-y-4">
                        {learning_resources && Object.keys(learning_resources).length > 0 ? (
                            Object.entries(learning_resources).map(([skill, resource], idx) => (
                                <div key={idx} className="bg-dark-800/50 p-4 rounded-lg border border-white/5">
                                    <h4 className="font-semibold text-purple-200 capitalize mb-1">{skill}</h4>
                                    <p className="text-sm text-gray-400">{resource}</p>
                                </div>
                            ))
                        ) : (
                            <p className="text-gray-500 italic">No specific resources found for missing skills.</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Analyzer;
