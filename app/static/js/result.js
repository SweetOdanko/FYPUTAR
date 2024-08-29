const urlParams = new URLSearchParams(window.location.search);
const image = urlParams.get('image');
const db_url = urlParams.get('db_url');
const error = urlParams.get('error');

const inputImage = document.getElementById('inputImage');
inputImage.src = image;

const imageId = image.split('/').pop();

if (error === 'No similar images detected') {
    displayNoSimilarImagesMessage();
} else {
    console.log(`Fetching similar images for: ${imageId}`);  
    fetch(`/api/get_similar_images?image=${encodeURIComponent(image)}&db_url=${encodeURIComponent(db_url)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displaySimilarImages(data.similar_images);
            } else {
                console.error('Error:', data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}

function displaySimilarImages(similarImages) {
    const similarImagesContainer = document.getElementById('similarImages');
    similarImagesContainer.innerHTML = ''; 

    if (similarImages.length === 0) {
        displayNoSimilarImagesMessage();
        return;
    }

    similarImages.forEach(imgSrc => {
        const imgElement = document.createElement('img');
        imgElement.src = `/api/get_image/${imgSrc}?db_url=${encodeURIComponent(db_url)}`;
        imgElement.style.maxWidth = '100%';
        imgElement.style.margin = '10px';
        imgElement.addEventListener('click', () => {
            openModal(`/api/get_image/${imgSrc}?db_url=${encodeURIComponent(db_url)}`);
        });

        similarImagesContainer.appendChild(imgElement);
    });
}

function displayNoSimilarImagesMessage() {
    const similarImagesContainer = document.getElementById('similarImages');
    similarImagesContainer.innerHTML = '<p>No similar images detected!</p>';
}

function openModal(imgSrc) {
    const modal = document.getElementById('download');
    const modalImage = document.getElementById('modalImage');
    const downloadButton = document.getElementById('downloadButton');
    modal.style.display = 'flex';
    modalImage.src = imgSrc;
    downloadButton.href = imgSrc;
}

const modal = document.getElementById('download');
const span = document.getElementById('closeModalButton');
span.onclick = function() {
    modal.style.display = 'none';
}
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

const downloadButton = document.getElementById('downloadButton');
downloadButton.onclick = function(event) {
    event.preventDefault();
    const link = document.createElement('a');
    link.href = downloadButton.href;
    link.download = '';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    modal.style.display = 'none';
    showCustomAlert();
}

function showCustomAlert() {
    const customAlert = document.getElementById('customAlert');
    customAlert.style.display = 'flex';
    const closeAlertButton = document.getElementById('closeAlertButton');
    closeAlertButton.onclick = function() {
        customAlert.style.display = 'none';
    }
}
