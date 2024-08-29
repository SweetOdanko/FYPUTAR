document.getElementById('upload-link').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function() {
    handleFiles(this.files);
    this.value = '';
});

function handleFiles(files) {
    if (files && files[0]) {
        let file = files[0];
        let formData = new FormData();
        formData.append('file', file);
        const dbUrl = 'mongodb://localhost:27017/';  // Adjust this to dynamically get the correct db_url
        formData.append('db_url', dbUrl);

        document.querySelector('.container').style.display = 'none';
        document.getElementById('loading').style.display = 'flex';

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            setTimeout(() => {
                document.getElementById('loading').style.display = 'none';
                document.querySelector('.container').style.display = 'block';
                if (data.success) {
                    window.location.href = `/results?image=${encodeURIComponent(data.image_url)}&db_url=${encodeURIComponent(dbUrl)}`;
                } else {
                    if (data.error === 'No similar images detected') {
                        showNoResultAlert("No similar images detected!");
                    } else if (data.error === 'Not supported image format!') {
                        showCustomAlert("Not supported image format!");
                    } else {
                        showCustomAlert("An error occurred during upload. Please try again.");
                    }
                }
            }, data.processing_time * 1000);
        })
        .catch(error => {
            console.error('Error:', error);
            showCustomAlert("An error occurred during upload. Please try again.");
        });
    }
}

function showCustomAlert(message) {
    const modal = document.getElementById('formatAlert');
    modal.querySelector('p').textContent = message;
    modal.style.display = 'flex';

    const closeModalButton = document.getElementById('closeModal-1');
    closeModalButton.removeEventListener('click', closeModal1);
    closeModalButton.addEventListener('click', closeModal1);
}

function closeModal1() {
    const modal = document.getElementById('formatAlert');
    modal.style.display = 'none';
}

function closeModal2() {
    const modal = document.getElementById('noResultAlert');
    modal.style.display = 'none';
}

function showNoResultAlert(message) {
    const modal = document.getElementById('noResultAlert');
    modal.querySelector('p').textContent = message;
    modal.style.display = 'flex';

    const closeModalButton = document.getElementById('closeModal-2');
    closeModalButton.removeEventListener('click', closeModal2);
    closeModalButton.addEventListener('click', closeModal2);
}

let dragArea = document.getElementById('drag-area');

dragArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dragArea.style.backgroundColor = '#f5f5f5';
});

dragArea.addEventListener('dragenter', (event) => {
    event.preventDefault();
    dragArea.style.backgroundColor = '#f5f5f5';
});

dragArea.addEventListener('dragleave', () => {
    dragArea.style.backgroundColor = '#eff1f3'; 
});

dragArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dragArea.style.backgroundColor = '#eff1f3'; 
    let files = event.dataTransfer.files;
    handleFiles(files);
});
