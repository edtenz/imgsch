// utils/image.js

export async function createThumbnail(dataUrl, maxSize) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            if (img.height > img.width) {
                canvas.height = maxSize;
                canvas.width = maxSize * (img.width / img.height);
            } else {
                canvas.width = maxSize;
                canvas.height = maxSize * (img.height / img.width);
            }

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            resolve(canvas.toDataURL());
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
}
