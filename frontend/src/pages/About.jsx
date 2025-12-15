import React from 'react';

const About = () => {
    return (
        <div className="max-w-3xl mx-auto space-y-12">
            <div className="glass-card p-10 rounded-2xl space-y-8">
                <div>
                    <h2 className="text-3xl font-bold mb-6">About The Project</h2>
                    <div className="space-y-4 text-gray-300 leading-relaxed">
                        <p>
                            In today's competitive job market, getting your resume past Applicant Tracking Systems (ATS)
                            is half the battle. This AI Resume Architect is designed to bridge the gap between
                            talented candidates and their dream jobs.
                        </p>
                        <p>
                            This project leverages state-of-the-art Natural Language Processing models to analyze
                            resumes just like a recruiter or an ATS would. By extracting keywords, analyzing sentiment,
                            and checking formatting compliance, we provide actionable insights to improve your hiralibity.
                        </p>
                        <p>
                            We believe every candidate deserves a fair chance. Our goal is to democratize
                            access to professional resume insights.
                        </p>
                    </div>
                </div>

                <div className="pt-8 border-t border-white/10">
                    <h3 className="text-xl font-bold mb-4">The Tech Stack</h3>
                    <ul className="grid grid-cols-2 gap-4 text-sm text-gray-400">
                        <li className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                            Frontend: React + Vite + TailwindCSS
                        </li>
                        <li className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
                            Backend: Python (FastAPI/Streamlit)
                        </li>
                        <li className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-green-500"></span>
                            AI/ML: Spacy, NLTK, Scikit-learn
                        </li>
                        <li className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                            Deployment: Docker & Cloud Containers
                        </li>
                    </ul>
                </div>

                <div className="text-center pt-8 text-xs text-gray-600">
                    Created with ❤️ for the AI Resume Analyzer Project
                </div>
            </div>
        </div>
    );
};

export default About;
