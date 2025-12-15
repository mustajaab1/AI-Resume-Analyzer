import React from 'react';
import { Scan, Database, BarChart3, Lock } from 'lucide-react';

const Features = () => {
    const features = [
        {
            title: "Dual-Engine Analysis",
            desc: "Choose between fast Keyword Matching (TF-IDF) or Deep Semantic Understanding (BERT + SBERT) for precise results.",
            icon: Scan,
            color: "text-blue-400"
        },
        {
            title: "Contextual Skill Matching",
            desc: "Beyond simple keyword spotting. Our Deep Analyzer understands the *context* of your skills using Sentence Transformers.",
            icon: Database,
            color: "text-purple-400"
        },
        {
            title: "Career & Gap Analysis",
            desc: "Identify exactly which skills you lack for your target role and get curated learning resources to bridge the gap.",
            icon: BarChart3,
            color: "text-emerald-400"
        },
        {
            title: "Privacy First Analysis",
            desc: "Your data is processed securely and ephemeral. We don't store your resume content after analysis.",
            icon: Lock,
            color: "text-pink-400"
        }
    ];

    return (
        <div className="space-y-12">
            <div className="text-center space-y-4">
                <h2 className="text-4xl font-bold">Powerful Features</h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                    Everything you need to optimize your resume for the modern job market.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                {features.map((feature, idx) => (
                    <div key={idx} className="glass-card p-8 rounded-2xl flex flex-col gap-4 hover:bg-dark-800/80 transition-colors group">
                        <div className={`p-4 rounded-xl bg-dark-900/50 w-fit ${feature.color} group-hover:scale-110 transition-transform`}>
                            <feature.icon size={32} />
                        </div>
                        <h3 className="text-xl font-bold">{feature.title}</h3>
                        <p className="text-gray-400 leading-relaxed">
                            {feature.desc}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Features;
