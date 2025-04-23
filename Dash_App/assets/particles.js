function createParticles() {
    const particles = document.querySelector('.particles');
    if (!particles) return;
    
    for(let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        Object.assign(particle.style, {
            width: `${Math.random() * 5 + 2}px`,
            height: particle.style.width,
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 20}s`
        });
        particles.appendChild(particle);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createParticles);
} else {
    createParticles();
}
