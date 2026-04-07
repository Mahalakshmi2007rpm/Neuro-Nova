function previewImage(event){
    document.getElementById("preview").src =
        URL.createObjectURL(event.target.files[0]);
}