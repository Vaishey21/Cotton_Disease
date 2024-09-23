document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('btnPredict').addEventListener('click', function() {
        var fileInput = document.getElementById('fileInput');
        var file = fileInput.files[0];
        
        if (!file) {
            alert('Please select an image.');
            return;
        }

        var formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            var resultDiv = document.getElementById('result');
            resultDiv.innerHTML = "<p>Predicted disease: " + data.predicted_disease + "</p>";
            resultDiv.innerHTML += "<p>Confidence level: " + data.confidence + "</p>";
        })
        .catch(error => console.error('Error:', error));
    });
});
