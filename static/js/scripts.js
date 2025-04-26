// static/js/scripts.js - Modified for better image handling and display

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const connectionStatus = document.getElementById('connection-status');
    const statusMessage = document.getElementById('status-message');
    const lastCaptureTime = document.getElementById('last-capture-time');
    const captureBtn = document.getElementById('capture-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const volumeInfoDiv = document.getElementById('volume-info');
    const volumeTableBody = document.querySelector('#volume-table tbody');
    const totalVolumeCell = document.getElementById('total-volume-cell');
    const downloadImageBtn = document.getElementById('download-image');
    const downloadCsvBtn = document.getElementById('download-csv');
    const downloadPdfBtn = document.getElementById('download-pdf');
    const processedImageLink = document.getElementById('processed-image-link');
    const processedImage = document.getElementById('processed-image');
    const modalImage = document.getElementById('modal-image');
    const imagePlaceholderIcon = document.getElementById('image-placeholder-icon');
    const imagePlaceholderText = document.getElementById('image-placeholder-text');
    const alertPlaceholder = document.getElementById('alert-placeholder');
    const settingsForm = document.getElementById('settings-form');
    const detectionThreshold = document.getElementById('detection-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const cameraEnabled = document.getElementById('camera-enabled');
    const refreshBtn = document.getElementById('refresh-btn');

    // Socket.IO setup
    const socket = io();
    let connected = false;
    
    // Image loading retry mechanism variables
    let imageRetryCount = 0;
    const MAX_IMAGE_RETRIES = 5;
    let imageRetryTimer = null;

    // Reset application state on page load - IMPORTANT FIX
    window.addEventListener('load', function() {
        console.log("Page loaded - resetting state");
        resetApplicationState();
        // Clear any cached data from sessionStorage
        sessionStorage.removeItem('lastImageData');
    
    // Add a periodic refresh for data when the connection is active
    // This ensures we get updates even if Socket.IO events fail
    setInterval(function() {
        if (connected && document.visibilityState !== 'hidden') {
            // Only refresh if we're connected and the page is visible
            console.log("Performing periodic data refresh");
            fetchLatestData();
        }
    }, 30000); // Refresh data every 30 seconds if connected
});

// Listen for visibility changes to refresh data when tab becomes visible again
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        console.log("Page became visible - refreshing data");
        // Small delay before refreshing
        setTimeout(function() {
            fetchLatestData();
        }, 500);
    }
});
    
    // Add refresh button functionality if it exists
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            if (!connected) {
                showAlert('Not connected to Raspberry Pi', 'warning');
                return;
            }
            
            // Show loading indicator
            document.querySelector('.image-placeholder').style.display = 'flex';
            imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-primary';
            imagePlaceholderText.textContent = 'Refreshing data...';
            
            // Reset image retry counter
            imageRetryCount = 0;
            if (imageRetryTimer) {
                clearTimeout(imageRetryTimer);
                imageRetryTimer = null;
            }
            
            // Request refresh
            fetchLatestData();
            
            // Also notify server through socket to get fresh data
            socket.emit('request_refresh');
        });
    }

    // Function to reset application state
    function resetApplicationState() {
        // Reset image display
        updateImageDisplay(null);
        
        // Reset volume table
        volumeTableBody.innerHTML = '';
        totalVolumeCell.textContent = 'Total Volume: 0.00 cc';
        
        // Hide volume info initially
        volumeInfoDiv.style.display = 'none';
        
        // Reset last capture time
        lastCaptureTime.textContent = 'None';
        
        // Show the image placeholder
        document.querySelector('.image-placeholder').style.display = 'flex';
        imagePlaceholderIcon.className = 'fas fa-camera fa-5x text-muted';
        imagePlaceholderText.textContent = 'No image available';
        
        // Reset image retry counter
        imageRetryCount = 0;
        if (imageRetryTimer) {
            clearTimeout(imageRetryTimer);
            imageRetryTimer = null;
        }
    }

    // Function to format timestamps
    function formatTimestamp(timestamp) {
        if (!timestamp) return 'None';
        
        // Check if timestamp is in YYYYMMDD-HHMMSS format
        if (/^\d{8}-\d{6}$/.test(timestamp)) {
            const year = timestamp.substring(0, 4);
            const month = timestamp.substring(4, 6);
            const day = timestamp.substring(6, 8);
            const hour = timestamp.substring(9, 11);
            const minute = timestamp.substring(11, 13);
            const second = timestamp.substring(13, 15);
            
            const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);
            return date.toLocaleString();
        }
        
        // If it's a different format, try to parse it directly
        return new Date(timestamp).toLocaleString();
    }

    // Function to update connection status
    function updateConnectionStatus(isConnected) {
        connected = isConnected;
        
        if (isConnected) {
            connectionStatus.className = 'alert connected';
            statusMessage.innerHTML = '<i class="fas fa-check-circle me-2"></i>Connected to Raspberry Pi';
            captureBtn.disabled = false;
            if (refreshBtn) refreshBtn.disabled = false;
        } else {
            connectionStatus.className = 'alert disconnected';
            statusMessage.innerHTML = '<i class="fas fa-exclamation-circle me-2"></i>Disconnected from Raspberry Pi';
            captureBtn.disabled = true;
            if (refreshBtn) refreshBtn.disabled = true;
        }
    }

    // Function to show alerts
    function showAlert(message, type='success', duration=5000) {
        const wrapper = document.createElement('div');
        wrapper.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        alertPlaceholder.appendChild(wrapper);
        
        // Auto-dismiss after duration
        if (duration > 0) {
            setTimeout(() => {
                wrapper.querySelector('.alert').classList.remove('show');
                setTimeout(() => {
                    wrapper.remove();
                }, 150);
            }, duration);
        }
    }

    // Function to clear alerts
    function clearAlerts() {
        alertPlaceholder.innerHTML = '';
    }

    // Populate the volume table with segments data
    function populateVolumeTable(segments, totalVolume) {
        // Clear existing rows
        volumeTableBody.innerHTML = '';

        // Add rows for each segment
        segments.forEach(seg => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${seg.section_id}</td>
                <td>${seg.volume_cc.toFixed(2)}</td>
                <td>${seg.width_cm.toFixed(2)}</td>
                <td>${seg.length_cm.toFixed(2)}</td>
                <td>${seg.height_cm.toFixed(2)}</td>
            `;
            volumeTableBody.appendChild(tr);
        });

        // Set total volume
        totalVolumeCell.textContent = `Total Volume: ${totalVolume.toFixed(2)} cc`;
    }

    // Function to update the image display - IMPROVED
    function updateImageDisplay(imageData) {
        // Clear any previous image
        processedImage.src = '';
        modalImage.src = '';
        
        if (!imageData) {
            // Hide image and show placeholder
            processedImageLink.style.display = 'none';
            document.querySelector('.image-placeholder').style.display = 'flex';
            imagePlaceholderIcon.className = 'fas fa-camera fa-5x text-muted';
            imagePlaceholderText.textContent = 'No image available';
            downloadImageBtn.disabled = true;
            return;
        }
        
        // Set image source
        const imgSrc = `data:image/jpeg;base64,${imageData}`;
        
        // Create a new image object to verify the base64 data is valid
        const testImg = new Image();
        testImg.onload = function() {
            // Image loaded successfully, update UI
            processedImage.src = imgSrc;
            modalImage.src = imgSrc;
            
            // Hide placeholder and show image
            document.querySelector('.image-placeholder').style.display = 'none';
            processedImageLink.style.display = 'block';
            downloadImageBtn.disabled = false;
            
            // Reset retry counter
            imageRetryCount = 0;
            if (imageRetryTimer) {
                clearTimeout(imageRetryTimer);
                imageRetryTimer = null;
            }
        };
        
        testImg.onerror = function() {
            console.error("Failed to load image from base64 data");
            document.querySelector('.image-placeholder').style.display = 'flex';
            imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
            imagePlaceholderText.textContent = 'Image data corrupted or invalid';
            processedImageLink.style.display = 'none';
            downloadImageBtn.disabled = true;
        };
        
        // Set the test image source to try loading
        testImg.src = imgSrc;
    }

    // Function to fetch and display the latest data - IMPROVED FOR REFRESH
    function fetchLatestData() {
        // Show loading indicator in placeholder
        document.querySelector('.image-placeholder').style.display = 'flex';
        imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-primary';
        imagePlaceholderText.textContent = 'Loading data...';
        
        fetch('/api/latest_data')
            .then(response => response.json())
            .then(data => {
                if (data.connected) {
                    updateConnectionStatus(true);
                    
                    // Update last capture time
                    lastCaptureTime.textContent = formatTimestamp(data.timestamp);
                    
                    // Update segments data if available
                    if (data.segments && data.segments.length > 0) {
                        populateVolumeTable(data.segments, data.total_volume);
                        volumeInfoDiv.style.display = 'block';
                    } else {
                        volumeInfoDiv.style.display = 'none';
                    }
                    
                    // Check if there's an image to fetch
                    if (data.has_image) {
                        fetchLatestImage();
                    } else {
                        // No image available
                        updateImageDisplay(null);
                        
                        // Check if we need to retry fetching the image
                        if (data.timestamp && imageRetryCount < MAX_IMAGE_RETRIES) {
                            imageRetryCount++;
                            console.log(`Image not yet available, retry ${imageRetryCount}/${MAX_IMAGE_RETRIES} in 2 seconds...`);
                            
                            imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-primary';
                            imagePlaceholderText.textContent = `Loading image... (try ${imageRetryCount}/${MAX_IMAGE_RETRIES})`;
                            
                            // Set a timeout to retry
                            if (imageRetryTimer) clearTimeout(imageRetryTimer);
                            imageRetryTimer = setTimeout(fetchLatestImage, 2000);
                        } else if (imageRetryCount >= MAX_IMAGE_RETRIES) {
                            // Max retries reached
                            imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                            imagePlaceholderText.textContent = 'Image not available after multiple attempts';
                            imageRetryCount = 0; // Reset for next time
                        }
                    }
                } else {
                    updateConnectionStatus(false);
                    updateImageDisplay(null);
                }
            })
            .catch(error => {
                console.error('Error fetching latest data:', error);
                updateConnectionStatus(false);
                updateImageDisplay(null);
                
                document.querySelector('.image-placeholder').style.display = 'flex';
                imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                imagePlaceholderText.textContent = 'Failed to load data';
            });
    }

    // Function to fetch the latest processed image - IMPROVED
    function fetchLatestImage() {
        // Update placeholder to show loading state
        document.querySelector('.image-placeholder').style.display = 'flex';
        imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-primary';
        imagePlaceholderText.textContent = 'Loading image...';
        
        fetch('/api/latest_image')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.image) {
                    console.log(`Received image data (length: ${data.image.length})`);
                    updateImageDisplay(data.image);
                } else {
                    console.warn("No image data in response");
                    updateImageDisplay(null);
                    
                    // Schedule a retry if needed
                    if (imageRetryCount < MAX_IMAGE_RETRIES) {
                        imageRetryCount++;
                        console.log(`Image still not available, retry ${imageRetryCount}/${MAX_IMAGE_RETRIES} in 2 seconds...`);
                        
                        if (imageRetryTimer) clearTimeout(imageRetryTimer);
                        imageRetryTimer = setTimeout(fetchLatestImage, 2000);
                    } else {
                        imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                        imagePlaceholderText.textContent = 'Image not available after multiple attempts';
                        imageRetryCount = 0; // Reset for next time
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching image:', error);
                
                if (imageRetryCount < MAX_IMAGE_RETRIES) {
                    imageRetryCount++;
                    console.log(`Error fetching image, retry ${imageRetryCount}/${MAX_IMAGE_RETRIES} in 2 seconds...`);
                    
                    imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-warning';
                    imagePlaceholderText.textContent = `Loading image... Error, retrying (${imageRetryCount}/${MAX_IMAGE_RETRIES})`;
                    
                    if (imageRetryTimer) clearTimeout(imageRetryTimer);
                    imageRetryTimer = setTimeout(fetchLatestImage, 2000);
                } else {
                    updateImageDisplay(null);
                    imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                    imagePlaceholderText.textContent = 'Failed to load image after multiple attempts';
                    imageRetryCount = 0; // Reset for next time
                }
            });
    }

    // Function to trigger a capture - IMPROVED FOR REFRESH
    function triggerCapture() {
        if (!connected) {
            showAlert('Not connected to Raspberry Pi', 'warning');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        captureBtn.disabled = true;
        
        // Clear previous image and show processing indicator
        updateImageDisplay(null);
        document.querySelector('.image-placeholder').style.display = 'flex';
        imagePlaceholderIcon.className = 'fas fa-sync fa-spin fa-5x text-primary';
        imagePlaceholderText.textContent = 'Processing image...';
        
        // Reset image retry counter
        imageRetryCount = 0;
        if (imageRetryTimer) {
            clearTimeout(imageRetryTimer);
            imageRetryTimer = null;
        }
        
        // Trigger image capture on the Raspberry Pi
        fetch('/api/trigger_capture', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            captureBtn.disabled = false;
            
            if (data.error) {
                showAlert(`Error: ${data.error}`, 'danger');
                imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                imagePlaceholderText.textContent = 'Capture failed';
            } else {
                showAlert('Image captured and analyzing...', 'success');
                
                // Set a timeout to fetch the latest data - using a slightly longer delay
                setTimeout(function() {
                    fetchLatestData();
                    
                    // Schedule additional refresh attempts to ensure we get the image
                    // Image processing might take time, so we'll retry a few times
                    for (let i = 1; i <= 3; i++) {
                        setTimeout(() => {
                            if (processedImageLink.style.display === 'none') {
                                console.log(`Scheduled refresh attempt ${i}`);
                                fetchLatestData();
                            }
                        }, 3000 * i);  // 3s, 6s, 9s
                    }
                    
                    // If no image is received after 15 seconds, reset the placeholder
                    setTimeout(function() {
                        if (processedImageLink.style.display === 'none') {
                            imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
                            imagePlaceholderText.textContent = 'Image processing timed out';
                            
                            // Add a manual refresh button
                            const refreshLink = document.createElement('a');
                            refreshLink.href = '#';
                            refreshLink.textContent = 'Click to refresh';
                            refreshLink.className = 'btn btn-sm btn-primary mt-3';
                            refreshLink.addEventListener('click', function(e) {
                                e.preventDefault();
                                fetchLatestData();
                            });
                            
                            // Clear existing content first
                            while (document.querySelector('.image-placeholder').lastChild) {
                                document.querySelector('.image-placeholder').removeChild(
                                    document.querySelector('.image-placeholder').lastChild
                                );
                            }
                            
                            // Add the new elements
                            document.querySelector('.image-placeholder').appendChild(imagePlaceholderIcon);
                            document.querySelector('.image-placeholder').appendChild(imagePlaceholderText);
                            document.querySelector('.image-placeholder').appendChild(refreshLink);
                        }
                    }, 15000);
                }, 3000);
            }
        })
        .catch(error => {
            console.error('Error triggering capture:', error);
            showAlert('Failed to communicate with the server', 'danger');
            loadingSpinner.style.display = 'none';
            captureBtn.disabled = false;
            imagePlaceholderIcon.className = 'fas fa-exclamation-circle fa-5x text-warning';
            imagePlaceholderText.textContent = 'Connection error';
        });
    }

    // Function to fetch settings
    function fetchSettings() {
        fetch('/api/get_config')
            .then(response => response.json())
            .then(data => {
                // Update form values
                if (data.detection_threshold) {
                    detectionThreshold.value = data.detection_threshold;
                    thresholdValue.textContent = data.detection_threshold;
                }
                
                if (data.camera_enabled !== undefined) {
                    cameraEnabled.checked = data.camera_enabled;
                }
            })
            .catch(error => {
                console.error('Error fetching settings:', error);
            });
    }

    // Download the processed image - IMPROVED
    downloadImageBtn.addEventListener('click', function() {
        if (!processedImage.src || processedImage.src === '' || 
            processedImage.src.indexOf('data:image') === -1) {
            showAlert('No processed image available to download', 'warning');
            return;
        }

        try {
            const link = document.createElement('a');
            link.href = processedImage.src;
            link.download = `rebar_analysis_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Error downloading image:', error);
            showAlert('Failed to download image: ' + error.message, 'danger');
        }
    });

    // Download volume data as CSV
    downloadCsvBtn.addEventListener('click', function() {
        const rows = document.querySelectorAll('#volume-table tbody tr');
        if (rows.length === 0) {
            showAlert('No volume data available to download', 'warning');
            return;
        }

        const headers = ['Segment No.', 'Volume (cc)', 'Width (cm)', 'Length (cm)', 'Height (cm)'];
        const csvRows = [headers.join(',')];

        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => cell.textContent);
            csvRows.push(rowData.join(','));
        });

        const totalText = totalVolumeCell.textContent.replace('Total Volume: ', '');
        csvRows.push(`Total,,,,${totalText}`);

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.href = url;
        link.download = `rebar_volume_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);