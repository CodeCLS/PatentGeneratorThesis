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
          if (!window.network) {
            attempts++;
            if (attempts < maxAttempts) {
              setTimeout(setupNetwork, 100);
            }
            return;
          }
          
          function extractEntityType(title) {
            if (!title) {
              return '';
            }
            const match = String(title).match(/type:\\s*([^<]+)/i);
            return match ? match[1].trim().toUpperCase() : '';
          }

          function extractEntityName(node) {
            if (!node || !node.options) {
              return '';
            }
            return node.options.label || '';
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
                    name: extractEntityName(node) || nodeId,
                    label: extractEntityType(node.options.title) || 'Unknown',
                  }, '*');
                }
              } else if (params.edges && params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = window.network.body.edges[edgeId];
                
                if (edge && edge.options) {
                  // Find triple index by searching through all edges
                  // In pyvis, edges are stored in network.body.data.edges
                  const edgeData = window.network.body.data.edges.get(edgeId);
                  const headNode = window.network.body.nodes[edge.fromId];
                  const tailNode = window.network.body.nodes[edge.toId];
                  
                  // Send message to parent window
                  window.parent.postMessage({
                    type: 'selectTriple',
                    index: edgeData.index !== undefined ? edgeData.index : -1,
                    relation: edge.options.label || '',
                    head: {
                      id: edge.fromId,
                      name: headNode ? extractEntityName(headNode) || edge.fromId : edge.fromId,
                      label: headNode ? extractEntityType(headNode.options && headNode.options.title) || 'Unknown' : 'Unknown',
                    },
                    tail: {
                      id: edge.toId,
                      name: tailNode ? extractEntityName(tailNode) || edge.toId : edge.toId,
                      label: tailNode ? extractEntityType(tailNode.options && tailNode.options.title) || 'Unknown' : 'Unknown',
                    },
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

