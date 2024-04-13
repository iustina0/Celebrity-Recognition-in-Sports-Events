function uploadVideo() {
    const videoInput = document.getElementById('videoInput');
    const processedVideo = document.getElementById('processedVideo');

    const videoFile = videoInput.files[0];
    if (!videoFile) {
        alert('Please select a video file.');
        return;
    }

    const formData = new FormData();
    formData.append('video', videoFile);

    fetch('http://localhost:3000/api/process-video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        console.log(blob);
        const url = URL.createObjectURL(blob);
        processedVideo.src = url;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}