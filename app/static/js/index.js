document.addEventListener('DOMContentLoaded', () => {
    const urlRadio = document.getElementById('urlRadio');
    const txtRadio = document.getElementById('txtRadio');
    const urlInput = document.querySelector('.url-input');
    const textInputs = document.querySelector('.text-inputs');

    function toggleInputs() {
        if (urlRadio.checked) {
            urlInput.classList.remove('d-none');
            textInputs.classList.add('d-none');
        } else if (txtRadio.checked) {
            urlInput.classList.add('d-none');
            textInputs.classList.remove('d-none');
        }
    }

    urlRadio.addEventListener('change', toggleInputs);
    txtRadio.addEventListener('change', toggleInputs);
    toggleInputs();
});
