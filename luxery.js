function exploreBrands() {
    alert("Explore the luxury brands at Phoenix Palladium!");
  }
  const slides = document.querySelectorAll('.slide');
        const controls = document.querySelectorAll('.slider-control');

        let currentSlide = 0;

        function showSlide(n) {
            slides.forEach((slide, index) => {
                slide.classList.remove('active');
                controls[index].classList.remove('active');
            });

            slides[n].classList.add('active');
            controls[n].classList.add('active');
            currentSlide = n;
        }

        controls.forEach(control => {
            control.addEventListener('click', () => {
                const slideIndex = parseInt(control.dataset.slide);
                showSlide(slideIndex);
            });
        });