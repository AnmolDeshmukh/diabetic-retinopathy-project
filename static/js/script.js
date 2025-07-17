// Add an event listener to the file input element with ID 'imageUpload'
document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0]; // Get the first selected file
    
    if (file) { // Check if a file was selected
        const reader = new FileReader(); // Create a new FileReader object
        
        // Define the onload event handler for the FileReader
        reader.onload = function(e) {
            const img = document.getElementById('uploaded-image'); // Get the image element by ID
            img.src = e.target.result; // Set the image source to the file's data URL
            img.style.display = 'block'; // Make sure the image is visible
        };
        
        reader.readAsDataURL(file); // Read the selected file as a Data URL
    }
});

// Add an event listener to the form with ID 'imageForm'
document.getElementById('imageForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission behavior
    
    // Placeholder for handling image upload logic
    alert('Image uploaded successfully!'); // Show an alert when the form is submitted
});
