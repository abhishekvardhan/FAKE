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

                        audioPlayer.src = data.audio_url;
                        audioPlayer.load();
                        const playPromise = audioPlayer.play();
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
                    button.innerText = "Record";
                    audioStream.getTracks().forEach(track => track.stop());

                };
                mediaRecorder.start();
                mediaRecorder.state = "recording";
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