document.addEventListener('DOMContentLoaded', () => {
    // Credit Score Validation
    const creditScore = document.getElementById('input-credit-score');
    if (creditScore) {
        creditScore.addEventListener('input', function() {
            const value = parseInt(this.value);
            if (value < 300 || value > 850) {
                this.classList.add('is-invalid');
                this.parentElement.classList.add('has-error');
            } else {
                this.classList.remove('is-invalid');
                this.parentElement.classList.remove('has-error');
            }
        });
    }

    // Age Validation
    const age = document.getElementById('input-age');
    if (age) {
        age.addEventListener('input', function() {
            const value = parseInt(this.value);
            if (value < 18 || value > 100) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }
});
