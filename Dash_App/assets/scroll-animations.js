window.addEventListener('scroll', () => {
    document.querySelectorAll('.prediction-card').forEach(card => {
        const cardTop = card.getBoundingClientRect().top;
        if (cardTop < window.innerHeight * 0.8) {
            card.classList.add('fade-in');
        }
    });
});
