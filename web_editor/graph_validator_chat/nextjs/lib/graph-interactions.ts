/**
 * Graph Interaction Utilities
 * Handles communication between iframe graph and parent page
 */

export function injectGraphClickHandler(iframeWindow: Window) {
  try {
    // Inject a script into the iframe that captures node/edge clicks
    const script = document.createElement('script');
    script.textContent = `
      (function() {
        // Wait for vis network to be ready
        let maxAttempts = 50;
        let attempts = 0;
        
        function setupNetwork() {
          if (typeof vis === 'undefined' || !window.network) {
            attempts++;
            if (attempts < maxAttempts) {
              setTimeout(setupNetwork, 100);
            }
            return;
          }
          
          // Add click event to the network
          window.network.on('click', function(params) {
            try {
              if (params.nodes && params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = window.network.body.nodes[nodeId];
                
                if (node && node.options) {
                  // Send message to parent window
                  window.parent.postMessage({
                    type: 'selectEntity',
                    id: nodeId,
                    name: node.options.title || nodeId,
                    label: node.options.label ? node.options.label.split('\\n')[0] : 'Unknown',
                  }, '*');
                }
              }
            } catch (err) {
              console.error('Error handling node click:', err);
            }
          });
          
          console.log('Graph click handler attached');
        }
        
        setupNetwork();
      })();
    `;
    
    if (iframeWindow.document.body) {
      iframeWindow.document.body.appendChild(script);
    } else if (iframeWindow.document.head) {
      iframeWindow.document.head.appendChild(script);
    }
  } catch (err) {
    console.error('Failed to inject graph click handler:', err);
  }
}

export function extractGraphData(iframeDoc: Document) {
  try {
    // Try to extract network data from the iframe
    const scripts = iframeDoc.querySelectorAll('script');
    let networkData: any = null;
    
    for (const script of scripts) {
      if (script.textContent && script.textContent.includes('new vis.Network')) {
        // Found network initialization script
        networkData = {
          hasNetwork: true,
        };
        break;
      }
    }
    
    return networkData;
  } catch (err) {
    console.error('Failed to extract graph data:', err);
    return null;
  }
}

