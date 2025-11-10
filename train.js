// Global variables
let detector; 
let video, uploadedMedia, canvas, ctx; 
let animationFrameId; 
let currentMediaElement; 
window.latestKeypoints = []; 

// --- Constants ---
const ALL_POSES = ["Tree", "Cobra", "Warrior", "DownwardDog", "Bridge", "Triangle"];
const poseDataset = {}; 
ALL_POSES.forEach(pose => poseDataset[pose] = []);

// MAPPING: Local image files for reference images (Requires 'images' folder)
const REFERENCE_IMAGE_URLS = {
    "Tree": "tree.jpg",
    "Cobra": "cobra.jpg",
    "Warrior": "images/warrior.jpg",
    "DownwardDog": "images/downward_dog.jpg",
    "Bridge": "images/bridge.jpg",
    "Triangle": "images/triangle.jpg"
};

// MoveNet COCO keypoint indices and connections for drawing
const MOVENET_CONNECTIONS = [
    [5, 6], [5, 11], [6, 12], [11, 12], 
    [5, 7], [7, 9], 
    [6, 8], [8, 10], 
    [11, 13], [13, 15], 
    [12, 14], [14, 16], 
    [0, 1], [0, 2], [1, 3], [2, 4] 
];

// --- Initialization ---
document.addEventListener('DOMContentLoaded', setup);

async function setup() {
    video = document.getElementById('webcam-input');
    uploadedMedia = document.getElementById('uploaded-media');
    canvas = document.getElementById('pose-canvas');
    ctx = canvas.getContext('2d');

    document.getElementById('loading-message').textContent = 'Loading MoveNet... This may take a moment.';
    try {
        const model = poseDetection.SupportedModels.MoveNet;
        const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING 
        };
        
        detector = await poseDetection.createDetector(model, detectorConfig);
        
        document.getElementById('loading-message').textContent = 'Model Loaded. Ready!';
        document.getElementById('save-pose-btn').disabled = false;
    } catch (error) {
        document.getElementById('loading-message').innerHTML = '<span class="error-message">Error loading MoveNet model. Check console for details.</span>';
        console.error("MoveNet Load Error:", error);
        return;
    }
    
    document.getElementById('mode-select').onchange = handleModeChange;
    document.getElementById('pose-select').onchange = updateReferenceImage;
    document.getElementById('file-upload').onchange = handleFileUpload;
    document.getElementById('save-pose-btn').onclick = capturePose;
    document.getElementById('export-data-btn').onclick = exportDataset;
    document.getElementById('pause-video-btn').onclick = toggleVideoPause;

    updateSampleCounts();
    updateReferenceImage();
    handleModeChange(); 
}

// --- Mode Switching Logic ---

async function handleModeChange() {
    const mode = document.getElementById('mode-select').value;
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    uploadedMedia.pause();

    video.style.display = 'none';
    uploadedMedia.style.display = 'none';
    canvas.style.display = 'block'; 
    
    document.getElementById('file-upload').style.display = 'none';
    document.getElementById('upload-label').style.display = 'none';
    document.getElementById('pause-video-btn').style.display = 'none';
    document.getElementById('loading-message').textContent = '';

    if (mode === 'camera') {
        await startCamera(); 
        document.getElementById('save-pose-btn').style.display = 'block';
        document.getElementById('pause-video-btn').style.display = 'none';
    } else if (mode === 'image' || mode === 'video') {
        document.getElementById('file-upload').accept = mode === 'image' ? 'image/*' : 'video/*';
        document.getElementById('file-upload').style.display = 'block';
        document.getElementById('upload-label').style.display = 'inline';
        document.getElementById('save-pose-btn').style.display = 'block';
        if (mode === 'video') {
            document.getElementById('pause-video-btn').style.display = 'block';
        }
    }
    
    canvas.width = 0; 
    canvas.height = 0;
}

// Robust startCamera function
async function startCamera() {
    document.getElementById('loading-message').textContent = 'Starting Camera...';
    
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        video.srcObject = stream;
        video.setAttribute('playsinline', true);
        video.setAttribute('autoplay', true);
        video.style.display = 'none'; 
        currentMediaElement = video;
        console.log("camera working fine");

        // Wait for video metadata to load and stream to stabilize
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                resolve(true); 
            };
        });

        await new Promise(r => setTimeout(r, 500)); 

        setupCanvas(video);
        detectAndDrawLoop(); 
        document.getElementById('loading-message').textContent = '';

    } catch (error) {
        document.getElementById('loading-message').innerHTML = 
            '<span class="error-message">Cannot access camera. Check browser permissions (HTTPS required) or if another app is using the camera.</span>';
        console.error("Camera Error:", error);
    }
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const mode = document.getElementById('mode-select').value;

    if (mode === 'image') {
        const img = new Image();
        img.onload = () => {
            uploadedMedia.style.display = 'none';
            setupCanvas(img);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            detectAndDrawStatic(img); 
            document.getElementById('loading-message').textContent = 'Image Ready. Save Pose.';
            URL.revokeObjectURL(img.src);
        };
        img.src = url;
        currentMediaElement = img;

    } else if (mode === 'video') {
        uploadedMedia.src = url;
        uploadedMedia.style.display = 'none';
        currentMediaElement = uploadedMedia;
        
        uploadedMedia.onloadedmetadata = () => {
            setupCanvas(uploadedMedia);
            detectAndDrawLoop(); 
            document.getElementById('loading-message').textContent = 'Video Ready. Press Play/Pause to capture frame.';
        };
    }
}

// MIRRORING AND VISIBILITY FIX: Sets CSS transform and explicit canvas dimensions
function setupCanvas(mediaElement) {
    // Use the actual dimensions reported by the media element
    const width = mediaElement.tagName === 'VIDEO' ? mediaElement.videoWidth : mediaElement.width;
    const height = mediaElement.tagName === 'VIDEO' ? mediaElement.videoHeight : mediaElement.height;
    
    // Set the canvas rendering size (internal resolution)
    canvas.width = width;
    canvas.height = height;
    
    // Set the canvas display size (CSS styles) - VISIBILITY FIX
    // Matches the fixed size set in the HTML's #input-container
    canvas.style.width = '640px'; 
    canvas.style.height = '480px';
    
    // Apply CSS transform to flip the canvas horizontally for mirror effect (camera mode only)
    if (mediaElement === video) {
        canvas.style.transform = 'scaleX(-1)'; 
    } else {
        canvas.style.transform = 'none';
    }
}

// --- Pose Detection Loop ---

async function detectAndDrawLoop() {
    if (!detector) {
        animationFrameId = requestAnimationFrame(detectAndDrawLoop);
        return;
    }
    
    if (currentMediaElement === uploadedMedia && uploadedMedia.paused) {
        animationFrameId = requestAnimationFrame(detectAndDrawLoop);
        return;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.drawImage(currentMediaElement, 0, 0, canvas.width, canvas.height);
    
    const poses = await detector.estimatePoses(currentMediaElement);

    if (poses && poses.length > 0) {
        const keypoints = poses[0].keypoints;
        
        drawSkeleton(keypoints, false); 
        
        window.latestKeypoints = []; 
        keypoints.forEach(kp => {
            const x_norm = kp.x / canvas.width;
            const y_norm = kp.y / canvas.height;
            
            // Data normalization uses raw coordinates
            const x_data = x_norm; 
            
            if (kp.score > 0.3) { 
                window.latestKeypoints.push(x_data, y_norm); 
            } else {
                window.latestKeypoints.push(-1, -1); 
            }
        });
    } else {
        window.latestKeypoints = [];
    }
    
    animationFrameId = requestAnimationFrame(detectAndDrawLoop);
}

// Detection for a static image (runs only once)
async function detectAndDrawStatic(imgElement) {
    if (!detector) return;
    
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

    const poses = await detector.estimatePoses(imgElement);

    window.latestKeypoints = [];

    if (poses && poses.length > 0) {
        const keypoints = poses[0].keypoints;
        drawSkeleton(keypoints, false); 
        
        keypoints.forEach(kp => {
            const x_norm = kp.x / canvas.width;
            const y_norm = kp.y / canvas.height;
            
            if (kp.score > 0.3) {
                window.latestKeypoints.push(x_norm, y_norm); 
            } else {
                window.latestKeypoints.push(-1, -1);
            }
        });
    }
}


// --- Utility and Control Functions ---

function toggleVideoPause() {
    const btn = document.getElementById('pause-video-btn');
    if (uploadedMedia.paused) {
        uploadedMedia.play();
        btn.textContent = '⏸️ Pause Video';
    } else {
        uploadedMedia.pause();
        btn.textContent = '▶️ Resume Video';
        if (currentMediaElement === uploadedMedia) {
             detectAndDrawStatic(uploadedMedia); 
        }
    }
}

function updateReferenceImage() {
    const pose = document.getElementById('pose-select').value;
    const refImg = document.getElementById('reference-image');
    
    if (REFERENCE_IMAGE_URLS[pose]) {
        refImg.src = REFERENCE_IMAGE_URLS[pose]; 
        refImg.alt = `Reference Image of ${pose} Pose`;
    } else {
        refImg.src = '';
        refImg.alt = 'Reference image not available';
    }
}

function updateSampleCounts() {
    const countDiv = document.getElementById('sample-counts');
    countDiv.innerHTML = '<h4>Samples Collected:</h4>';
    let totalSamples = 0;
    
    for (const pose in poseDataset) { // FIX: Corrected syntax to for...in
        totalSamples += poseDataset[pose].length;
        countDiv.innerHTML += `<div class="pose-count"><strong>${pose}:</strong> ${poseDataset[pose].length} samples</div>`;
    }
    if (totalSamples === 0) {
        countDiv.innerHTML += '<div>Start capturing data!</div>';
    }
}

function capturePose() {
    const selectedPose = document.getElementById('pose-select').value;
    const keypointsToSave = window.latestKeypoints;
    
    if (keypointsToSave.length !== 34) { 
        alert(`Error: Keypoints not ready or pose detection failed. Expected 34 features, got ${keypointsToSave.length}.`);
        return;
    }

    poseDataset[selectedPose].push(keypointsToSave);
    updateSampleCounts();
    console.log(`Saved 1 sample for ${selectedPose}. Total: ${poseDataset[selectedPose].length}`);
}

function exportDataset() {
    const allData = [];
    
    for (const pose in poseDataset) { // FIX: Corrected syntax to for...in
        if (poseDataset[pose].length > 0) {
            allData.push({
                "pose": pose,
                "samples": poseDataset[pose]
            });
        }
    }
    
    if (allData.length === 0) {
        alert("No data collected yet!");
        return;
    }

    const filename = 'yoga_training_data.json';
    const jsonString = JSON.stringify(allData, null, 2);
    
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// --- Drawing Function ---

function drawSkeleton(keypoints, isMirrored) {
    // Draw points
    keypoints.forEach(keypoint => {
        if (keypoint.score > 0.3) {
            const x_draw = keypoint.x;
            const y_draw = keypoint.y;

            ctx.beginPath();
            ctx.arc(x_draw, y_draw, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#4CAF50'; 
            ctx.fill();
        }
    });

    // Draw lines (bones)
    MOVENET_CONNECTIONS.forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];

        if (kp1.score > 0.3 && kp2.score > 0.3) {
            const x1 = kp1.x;
            const y1 = kp1.y;
            const x2 = kp2.x;
            const y2 = kp2.y;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}