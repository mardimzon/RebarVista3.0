// static/js/welcome.js

document.addEventListener('DOMContentLoaded', function() {
    // Help toggle functionality
    const helpToggle = document.getElementById('help-toggle');
    const helpContent = document.getElementById('help-content');
    
    if (helpToggle && helpContent) {
        helpToggle.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Toggle the help content visibility
            if (helpContent.style.display === 'none' || helpContent.style.display === '') {
                helpContent.style.display = 'block';
                helpToggle.innerHTML = '<i class="fas fa-times-circle"></i> Hide help';
            } else {
                helpContent.style.display = 'none';
                helpToggle.innerHTML = '<i class="fas fa-question-circle"></i> Need help connecting?';
            }
        });
    }
    
    // Pre-check connection status
    checkConnectionStatus();
    
    function checkConnectionStatus() {
        // Try to fetch connection status to see if the server is running
        fetch('/api/connection_status')
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Network response was not ok');
            })
            .then(data => {
                // If we can connect to the server, check if it's connected to the Raspberry Pi
                if (data.connected) {
                    // Add a connected badge to the UI
                    const connectionBadge = document.createElement('div');
                    connectionBadge.className = 'connection-badge connected';
                    connectionBadge.innerHTML = '<i class="fas fa-check-circle"></i> Raspberry Pi Connected';
                    
                    // Add it above the start button
                    const startButton = document.querySelector('.start-button');
                    startButton.parentNode.insertBefore(connectionBadge, startButton);
                } else {
                    // Add a warning badge
                    const connectionBadge = document.createElement('div');
                    connectionBadge.className = 'connection-badge warning';
                    connectionBadge.innerHTML = '<i class="fas fa-exclamation-circle"></i> Raspberry Pi Not Detected';
                    
                    // Add it above the start button
                    const startButton = document.querySelector('.start-button');
                    startButton.parentNode.insertBefore(connectionBadge, startButton);
                }
            })
            .catch(error => {
                console.log('Error checking connection status:', error);
                // We don't need to show an error on the welcome page
                // The main page will handle connection issues
            });
    }
    
    // Add connection badge styles dynamically
    addConnectionBadgeStyles();
    
    function addConnectionBadgeStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .connection-badge {
                display: inline-block;
                padding: 8px 15px;
                border-radius: 20px;
                margin-bottom: 20px;
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            .connection-badge.connected {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .connection-badge.warning {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            
            .connection-badge i {
                margin-right: 5px;
            }
        `;
        document.head.appendChild(style);
    }
});