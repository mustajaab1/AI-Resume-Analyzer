import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, FileText, AlertTriangle, Check, X, Download } from 'lucide-react';

const Analyzer = () => {
    const [loading, setLoading] = useState(true);
    const location = useLocation();
    const fileName = location.state?.fileName || "Resume.pdf";

    useEffect(() => {
        // Simulate processing time
        const timer = setTimeout(() => {
            setLoading(false);
        }, 3000);
        return () => clearTimeout(timer);
    }, []);

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

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6"
        >
            <header className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-2xl font-bold">Analysis Report</h1>
                    <p className="text-gray-400 text-sm">Target: General Software Engineering</p>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-dark-800 rounded-lg hover:bg-dark-700 border border-white/10 transition-colors">
                    <Download size={16} /> Export PDF
                </button>
            </header>

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
                                stroke="#22c55e"
                                strokeWidth="10"
                                strokeDasharray="283"
                                strokeDashoffset="42" // 85% roughly
                                strokeLinecap="round"
                                className="drop-shadow-[0_0_10px_rgba(34,197,94,0.5)]"
                            />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-4xl font-bold">85</span>
                            <span className="text-xs text-gray-400 uppercase tracking-widest mt-1">Score</span>
                        </div>
                    </div>
                </div>

                {/* Baseline Summary */}
                <div className="glass-card p-6 rounded-2xl lg:col-span-2 space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                        <FileText className="text-primary" size={20} />
                        <h3 className="text-lg font-semibold">Baseline Summary</h3>
                    </div>
                    <p className="text-gray-300 leading-relaxed">
                        Your resume shows strong potential for Software Engineering roles.
                        The structure is clean, and you have highlighted key technical skills effectively.
                        However, there are opportunities to improve impact metrics in your experience section
                        and align your terminology more closely with industry-standard job descriptions
                        to maximize ATS pass rates.
                    </p>
                </div>
            </div>

            {/* Bottom Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

                {/* Detected Keywords */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-blue-400">Detected Keywords & Skills</h3>
                    <div className="flex flex-wrap gap-2">
                        {['Python', 'React', 'Node.js', 'AWS', 'Docker', 'Machine Learning', 'API Design', 'Git'].map(tag => (
                            <span key={tag} className="px-3 py-1 rounded-full text-sm bg-blue-500/10 text-blue-300 border border-blue-500/20">
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Recommended Keywords */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-orange-400">Recommended / Missing</h3>
                    <div className="flex flex-wrap gap-2">
                        {['CI/CD', 'Kubernetes', 'System Design', 'Agile', 'GraphQL', 'Unit Testing'].map(tag => (
                            <span key={tag} className="px-3 py-1 rounded-full text-sm bg-orange-500/10 text-orange-300 border border-orange-500/20 dashed-border">
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Critical Formatting */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-red-400">Critical Issues</h3>
                    <ul className="space-y-3">
                        {[
                            'Contact info missing from header',
                            'Inconsistent date formatting',
                            'Missing LinkedIn profile',
                            'Section headers not standardized'
                        ].map((issue, idx) => (
                            <li key={idx} className="flex items-start gap-3 text-sm text-gray-300">
                                <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                                {issue}
                            </li>
                        ))}
                    </ul>
                </div>

            </div>
        </motion.div>
    );
};

export default Analyzer;
