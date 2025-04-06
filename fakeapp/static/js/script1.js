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
    }, 300);

    let button = document.getElementById("controlButton");
    let audioPlayer = document.getElementById("audio_player");
    let displayText = document.getElementById("displayText");
    let audioFileUrl = button.dataset.audioUrl;
    let mediaRecorder;
    let audioChunks = [];
    let audioStream;

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
            displayText.innerText = "Tell Me about yourself!";
            button.innerText = "Record";
        } else if (button.innerText === "Record") {
            try {
                // Reset audio chunks array before starting a new recording
                audioChunks = [];

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

                // Create media recorder with appropriate MIME type
                // Use audio/webm;codecs=opus for better quality and compatibility
                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : (MediaRecorder.isTypeSupported('audio/webm')
                        ? 'audio/webm'
                        : 'audio/mp4');

                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType,
                    audioBitsPerSecond: 128000 // Higher bitrate for better quality
                });

                // Set up data handling - collect ALL audio data
                mediaRecorder.ondataavailable = event => {
                    if (event.data && event.data.size > 0) {
                        audioChunks.push(event.data);
                        console.log("Audio chunk received:", event.data.size, "bytes");
                    }
                };

                // Log when recording starts
                mediaRecorder.onstart = () => {
                    console.log("MediaRecorder started", mediaRecorder.state);
                    displayText.innerText = "Recording in progress...";
                    // Visual indicator for recording
                    displayText.style.color = "red";
                };

                // Start recording with smaller chunks (100ms) to ensure we capture everything
                mediaRecorder.start(100);
                button.innerText = "Stop Record";

            } catch (err) {
                console.error("Error starting recording:", err);
                alert("Failed to start recording: " + err.message);
                button.innerText = "Record";
            }
        } else if (button.innerText === "Stop Record") {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                try {
                    // Set up onstop handler BEFORE calling stop
                    mediaRecorder.onstop = async () => {
                        try {
                            console.log("MediaRecorder stopped", mediaRecorder.state);
                            console.log("Audio chunks collected:", audioChunks.length);

                            if (audioChunks.length === 0) {
                                console.error("No audio data recorded");
                                alert("No audio data recorded. Please try again.");
                                button.innerText = "Record";
                                displayText.style.color = "";
                                return;
                            }

                            // Create blob from audio chunks
                            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                            console.log("Audio blob created:", audioBlob.size, "bytes");

                            // Debug: Play locally to verify recording
                            const audioUrl = URL.createObjectURL(audioBlob);
                            console.log("Local audio URL created:", audioUrl);

                            // Create form data
                            const formData = new FormData();
                            formData.append("audio", audioBlob, "recording." + (mediaRecorder.mimeType.includes("webm") ? "webm" : "mp4"));

                            // Send to server
                            console.log("Sending audio to server...");
                            const response = await fetch("/upload-audio/", {
                                method: "POST",
                                body: formData
                            });

                            if (!response.ok) {
                                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                            }

                            const data = await response.json();
                            console.log("Server response:", data);

                            // Update UI based on response
                            if (data.audio_url) {
                                audioPlayer.src = data.audio_url;
                                displayText.innerText = data.text;
                                displayText.style.color = "";
                                button.innerText = data.button_text;
                            } else {
                                button.innerText = "Record";
                                displayText.innerText = "No response received. Try again.";
                                displayText.style.color = "";
                            }

                            if (data.is_last) {
                                button.disabled = true;
                            }

                            // Stop all tracks in the audio stream to release the microphone
                            if (audioStream) {
                                audioStream.getTracks().forEach(track => track.stop());
                            }

                        } catch (err) {
                            console.error("Error processing recording:", err);
                            alert("Error processing recording: " + err.message);
                            button.innerText = "Record";
                            displayText.style.color = "";
                        }
                    };

                    // Now stop the recorder
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
    });
});