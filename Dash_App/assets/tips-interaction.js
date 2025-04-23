document.addEventListener('DOMContentLoaded', () => {
    // Tip Card Interactions
    document.querySelectorAll('.tip-btn').forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            document.querySelectorAll('.tip-btn').forEach(btn => {
                btn.classList.remove('active-tip');
            });
            
            // Add active class to clicked button
            this.classList.add('active-tip');
            
            // Animate tip content
            const tipContent = document.getElementById('dynamic-tips-content');
            tipContent.style.opacity = 0;
            setTimeout(() => {
                tipContent.style.opacity = 1;
            }, 300);
        });
    });

    // Hover effects for cards
    document.querySelectorAll('.comparison-card, .prediction-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'perspective(1000px) rotateX(2deg) rotateY(2deg) scale(1.02)';
            card.style.boxShadow = '0 12px 24px rgba(25, 34, 49, 0.2)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'none';
            card.style.boxShadow = '0 4px 6px rgba(25, 34, 49, 0.1)';
        });
    });
});
