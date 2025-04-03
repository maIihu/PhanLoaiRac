const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const croppedImagesDiv = document.getElementById("croppedImages");

fileInput.addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        clearOldImages();
        result.innerText = "Chưa có kết quả";

        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});

function clearOldImages() {
    let existingImage = document.getElementById("selected-image");
    if (existingImage) {
        existingImage.remove();
    }
    console.log("Requesting to clear old images...");
    fetch("/clear_output_folder", {
        method: "GET",
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Ảnh cũ đã được xóa.");
            croppedImagesDiv.innerHTML = '';
        } else {
            console.error("Lỗi khi xóa ảnh cũ:", data.error);
        }
    })
    .catch(error => console.error("Lỗi khi gửi yêu cầu xóa ảnh:", error));
}

function uploadImage() {
    const file = fileInput.files[0];
    if (!file) {
        alert("Vui lòng chọn ảnh!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/detect_objects", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi: " + data.error);
        } else {
            displayCroppedImages(data.cropped_images);
        }
    })
    .catch(error => console.error("Lỗi:", error));
}

function displayCroppedImages(images) {
    croppedImagesDiv.innerHTML = '';
    images.forEach(image => {
        const container = document.createElement("div");
        container.className = "cropped-image-container";

        const img = document.createElement("img");
        const uniqueImageUrl = `${image}?${new Date().getTime()}`;
        img.src = uniqueImageUrl;
        img.className = "cropped-image";

        container.onclick = () => showInRightPanel(uniqueImageUrl, image);

        container.appendChild(img);
        croppedImagesDiv.appendChild(container);
    });
}

function showInRightPanel(imageUrl, imagePath) {
    const rightPanel = document.querySelector(".right-panel");
    const resultElement = document.getElementById("result").parentElement;

    let existingImage = document.getElementById("selected-image");
    if (existingImage) {
        existingImage.remove();
    }

    const img = document.createElement("img");
    img.id = "selected-image";
    img.src = imageUrl;

    rightPanel.insertBefore(img, resultElement);

    classifyImage(imagePath);
}

async function classifyImage(imageUrl) {
    try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const file = new File([blob], "object.jpg", { type: "image/jpeg" });

        const formData = new FormData();
        formData.append("file", file);

        const predictResponse = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await predictResponse.json();
        result.innerText = `Loại: ${data.predicted_class} (${data.confidence.toFixed(2)}%)`;
    } catch (error) {
        console.error("Lỗi:", error);
    }
}
