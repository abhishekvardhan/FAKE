document.addEventListener("DOMContentLoaded", async function () {
    // Load models from the Django static weights folder
    await faceapi.nets.tinyFaceDetector.loadFromUri('/static/weights/');
    // await faceapi.nets.faceLandmark68Net.loadFromUri('/static/weights/');
    // await faceapi.nets.faceRecognitionNet.loadFromUri('/static/weights/');

    // Access webcam
    const video = document.getElementById("video");
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((err) => console.error("Error accessing webcam:", err));

    // Detect faces in real-time
    const canvas = document.getElementById("overlay");
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);
    let noFaceTimer = null;
    let noFaceStartTime = null;
    const noFaceTimeLimit = 3000; // 5 seconds in milliseconds
    let popupShown = false;

    // Create popup element
    const popup = document.createElement("div");
    popup.style.position = "fixed";
    popup.style.top = "50%";
    popup.style.left = "50%";
    popup.style.transform = "translate(-50%, -50%)";
    popup.style.background = "rgba(255, 0, 0, 0.8)";
    popup.style.color = "white";
    popup.style.padding = "20px";
    popup.style.borderRadius = "5px";
    popup.style.zIndex = "1000";
    popup.style.display = "none";
    popup.textContent = "Face not detected! Please position yourself in front of the camera.";

    // Add close button
    const closeButton = document.createElement("button");
    closeButton.textContent = "OK";
    closeButton.style.marginTop = "10px";
    closeButton.style.padding = "5px 10px";
    closeButton.style.display = "block";
    closeButton.style.margin = "10px auto 0";
    closeButton.addEventListener("click", function () {
        popup.style.display = "none";
        popupShown = false;
    });

    popup.appendChild(closeButton);
    document.body.appendChild(popup);

    setInterval(async () => {
        // Detect faces
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
        console.log(detections);

        // Resize detections
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Clear canvas
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw face detections
        faceapi.draw.drawDetections(canvas, resizedDetections);
        // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        if (detections.length === 0) {
            // No face detected
            if (noFaceStartTime === null) {
                // Start the timer
                noFaceStartTime = Date.now();
            } else {
                // Check if 5 seconds has passed
                const timeElapsed = Date.now() - noFaceStartTime;
                if (timeElapsed >= noFaceTimeLimit && !popupShown) {
                    // Show popup after 5 seconds
                    popup.style.display = "block";
                    popupShown = true;
                }
            }
        } else {
            // Face detected, reset the timer
            noFaceStartTime = null;
            if (popupShown) {
                popup.style.display = "none";
                popupShown = false;
            }
        }
    }, 300);

    let button = document.getElementById("controlButton");
    let audioPlayer = document.getElementById("audio_player");
    let displayText = document.getElementById("displayText");
    let questionTitle = document.getElementById("displayquestion");
    let audioFileUrl = button.dataset.audioUrl;
    let mediaRecorder;
    let audioChunks = [];
    let audioStream;
    let myserial;
    let timerInterval;
    let timeLeft = 60;  // 60 seconds for recording
    const timerDisplay = document.getElementById("timerDisplay");
    //  show questio-title class  controlButton is not equal to "Start" or "Finish" 


    // List available audio devices for debugging
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            devices.forEach((device, index) => {
                if (device.kind === "audioinput") {
                    console.log(`Index ${index} | Label: ${device.label} | ID: ${device.deviceId}`);
                }
            });
        })
        .catch(err => console.error("Error listing devices:", err));

    button.addEventListener("click", async function () {
        if (button.innerText === "Start") {
            if (!audioFileUrl || audioFileUrl.trim() === "") {
                console.error("Audio file URL is null or empty.");
                alert("Error: No audio file available!");
                return;
            }
            console.log("Audio file URL:", audioFileUrl);
            audioPlayer.src = audioFileUrl;
            audioPlayer.load();
            audioPlayer.play();
            questionTitle.innerText = "Question";
            displayText.innerText = "Tell Me about yourself!";
            button.innerText = "Record";
        } else if (button.innerText === "Record") {
            try {
                
                // Request microphone access with specific constraints for better audio quality
                audioStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 44100,
                        channelCount: 1
                    }
                });
                mediaRecorder = new MediaRecorder(audioStream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioBuffer = await new AudioContext().decodeAudioData(arrayBuffer);
                    const wavBlob = encodeWAV(audioBuffer);
                    const wavUrl = URL.createObjectURL(wavBlob);
                    clearInterval(timerInterval);
                    timerDisplay.innerText = "";
                    audioPlayer.src = wavUrl;
                    const formData = new FormData();
                    console.log("WAV Blob size:", wavBlob.size);
                    formData.append("audio", wavBlob, "recording.wav");
                    try {
                        const response = await fetch("/upload-audio/", {
                            method: "POST",
                            body: formData

                        });

                        const data = await response.json();
                        displayText.innerText = data.text;
                        console.log(data.audio_url);
                        myserial = data.session_id;
                        console.log("Session ID:", myserial);
                        audioPlayer.src = data.audio_url;
                        audioPlayer.load();
                        const playPromise = audioPlayer.play();
                        button.innerText = data.button_text;
                        if (playPromise !== undefined) {
                            playPromise
                                .then(() => console.log("Audio playback started successfully"))
                                .catch(error => {
                                    console.error("Playback prevented by browser:", error);
                                    // Add a button for the user to click to start playback
                                    const playButton = document.createElement("button");
                                    playButton.innerText = "Play Response";
                                    playButton.onclick = () => audioPlayer.play();
                                    document.body.appendChild(playButton);
                                });
                        }
                        console.log("Response data:", data.audio_url);


                    } catch (error) {
                        console.error("Error uploading audio:", error);
                        displayText.innerText = "Error uploading audio.";
                        displayText.style.color = "red";
                    }

                    audioStream.getTracks().forEach(track => track.stop());

                };
                mediaRecorder.start();
                mediaRecorder.state = "recording";
                timeLeft = 60;
timerDisplay.innerText = `Recording: ${timeLeft}s`;

timerInterval = setInterval(() => {
    timeLeft--;
    timerDisplay.innerText = `Recording: ${timeLeft}s`;
    if (timeLeft <= 0) {
        clearInterval(timerInterval);
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            button.innerText = "Processing...";
        }
    }
}, 1000);
                // Create media recorder with appropriate MIME type
                // Use audio/webm;codecs=opus for better quality and compatibility
                button.innerText = "Stop Record";
            } catch (err) {
                console.error("Error starting recording:", err);
                alert("Failed to start recording: " + err.message);
                button.innerText = "Record";
            }
        } else if (button.innerText === "Stop Record") {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                try {
                    clearInterval(timerInterval);
timerDisplay.innerText = "";
                    // Set up onstop handler BEFORE calling stop
                    mediaRecorder.stop();


                    button.innerText = "Processing...";


                } catch (err) {
                    console.error("Error stopping recording:", err);
                    button.innerText = "Record";
                    displayText.style.color = "";
                }
            } else {
                console.error("MediaRecorder not in recording state:", mediaRecorder ? mediaRecorder.state : "undefined");
                button.innerText = "Record";
                displayText.style.color = "";
            }
        }
        else if (button.innerText === "Finish") {
            console.log("Finishing process...");
            questionTitle.innerText = "You sucessfully completed the interview!";
            displayText.innerText = "Thank you for your time!";
            try {
                const formData = new FormData();
                formData.append("serial", myserial);
                console.log("Session ID sent:", formData.get("serial"));
                const response = await fetch("/result/", {
                    method: "POST",
                    body: formData

                });
                console.log("Response ", response.status);
                const data = await response.json();
                if (data.redirect_url) {
                    window.location.href = data.redirect_url;
                } else {
                    console.error("Redirect URL not received.");
                }
            }
            catch (error) {
                console.error("Error fetching result:", error);

            }
        }
    });
    function encodeWAV(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const samples = mergeBuffers(audioBuffer, bitDepth);
        const byteRate = sampleRate * numChannels * bitDepth / 8;
        const blockAlign = numChannels * bitDepth / 8;
        const dataSize = samples.length * (bitDepth / 8);

        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, 'WAVE');

        // fmt sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
        view.setUint16(20, format, true); // AudioFormat
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);

        // data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        // Write samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        console.log("Encoded WAV length:", view.byteLength);
        return new Blob([view], { type: 'audio/wav' });
    }

    function mergeBuffers(audioBuffer, bitDepth) {
        const channels = [];
        for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
            channels.push(audioBuffer.getChannelData(i));
        }

        const length = channels[0].length;
        const result = new Float32Array(length * channels.length);

        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < channels.length; channel++) {
                result[i * channels.length + channel] = channels[channel][i];
            }
        }
        console.log("Merged buffer length:", result.length);
        return result;
    }
});