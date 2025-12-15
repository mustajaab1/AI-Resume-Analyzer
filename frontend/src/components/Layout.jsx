import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { FileText, Github } from 'lucide-react';

const Layout = () => {
    const location = useLocation();

    const isActive = (path) => {
        return location.pathname === path ? 'text-primary-glow font-semibold' : 'text-gray-400 hover:text-white transition-colors';
    };

    return (
        <div className="min-h-screen bg-dark-900 text-white selection:bg-primary-glow/30 selection:text-white">
            {/* Background Ambience */}
            <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-primary/20 rounded-full blur-[100px] opacity-20 animate-blob"></div>
                <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-secondary/20 rounded-full blur-[100px] opacity-20 animate-blob animation-delay-2000"></div>
                <div className="absolute top-[40%] left-[40%] w-[400px] h-[400px] bg-accent/10 rounded-full blur-[100px] opacity-20 animate-blob animation-delay-4000"></div>
            </div>

            <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-dark-900/80 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 group">
                        <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary/20 transition-colors">
                            <FileText size={24} />
                        </div>
                        <span className="font-bold text-xl tracking-tight text-white group-hover:text-primary-glow transition-colors">
                            AI Resume Analyzer
                        </span>
                    </Link>

                    <div className="flex items-center gap-8">
                        <ul className="hidden md:flex items-center gap-8 text-sm font-medium">
                            <li><Link to="/" className={isActive('/')}>Home</Link></li>
                            <li><Link to="/features" className={isActive('/features')}>Features</Link></li>
                            <li><Link to="/about" className={isActive('/about')}>About</Link></li>
                        </ul>
                        {/* Optional: Add a Github link or CTA if visible in original, though not explicitly requested in Nav */}
                    </div>
                </div>
            </nav>

            <main className="relative z-10 pt-24 pb-12 px-6 max-w-7xl mx-auto min-h-[calc(100vh-64px)]">
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
