var video = document.querySelector("#videoElement");

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
        })
        .catch(function (err0r) {
            console.log("Something went wrong!");
        });
}


async function app() {
    console.log('Loading model..');
    net = await tf.automl.loadImageClassification('model.json');
    console.log('Successfully loaded model');

    const webcam = await tf.data.webcam(webcamElement);
    while (isPredicting) {
        const img = await webcam.capture();
        const result = await net.classify(img);

        console.log(result);

        document.getElementById("predictions-mask").innerText = result;
        img.dispose();

        await tf.nextFrame();
    }
}   