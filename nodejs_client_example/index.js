require("dotenv").config();
const axios = require("axios");
const fs = require("fs");
const path = require("path");

// Directory to save the images
const saveDirectory = "./images";

async function fetchAndSaveImages(imageUrl, nameOfFile) {
    const url = `${process.env.BASE_URL}?url=${imageUrl}`;
    try {
        // Make a GET request to fetch the response data
        const response = await axios.get(url);

        // Get the response data
        const { masks, image_base64 } = response.data;

        // Ensure the save directory exists
        if (!fs.existsSync(saveDirectory)) {
            fs.mkdirSync(saveDirectory);
        }

        // Function to save a base64 image with a given file name
        const saveBase64Image = (base64Image, fileName) => {
            const filePath = path.join(saveDirectory, fileName);
            const base64Data = base64Image.replace(
                /^data:image\/\w+;base64,/,
                ""
            );
            const binaryData = Buffer.from(base64Data, "base64");
            fs.writeFileSync(filePath, binaryData);
            console.log(`Image saved as ${fileName} at ${filePath}`);
        };

        // Count existing images in the directory to determine file numbering
        const existingFiles = fs.readdirSync(saveDirectory);
        let imageCount = existingFiles.length + 1;

        // Save the main image
        saveBase64Image(image_base64, `${nameOfFile}${imageCount}.jpg`);
        imageCount++;

        // Save each mask image
        masks.forEach((maskBase64, index) => {
            saveBase64Image(maskBase64, `${nameOfFile}_mask_${index}.jpg`);
        });
    } catch (error) {
        console.error("Error fetching and saving images:", error);
    }
}

const imageUrl =
    "https://cdn.britannica.com/09/241709-050-149181B1/apple-iphone-11-2019.jpg";
const nameOfFile = "Iphone";
fetchAndSaveImages(imageUrl, nameOfFile);
