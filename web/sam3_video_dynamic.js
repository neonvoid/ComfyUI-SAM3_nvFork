/**
 * SAM3 Video Segmentation Dynamic Widget Management
 *
 * Toggles the text_prompt widget visibility based on prompt_mode.
 * Input slots are always visible — splicing node.inputs breaks
 * connection slot indices on workflow reload (causes "tuple index
 * out of range" validation errors that require recreating the node).
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false;

function log(...args) {
    if (DEBUG) {
        console.log("[SAM3-Video]", ...args);
    }
}

// Helper function to hide a widget (safe — widgets don't carry connections)
function hideWidget(node, widget) {
    if (!widget) return;
    if (widget._hidden) return;

    const index = node.widgets.indexOf(widget);
    if (index === -1) return;

    // Store original properties for restoration
    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    widget._originalIndex = index;
    widget._hidden = true;

    // Remove widget from array
    node.widgets.splice(index, 1);

    // Hide linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w));
    }
}

// Helper function to show a widget
function showWidget(node, widget) {
    if (!widget) return;
    if (!widget._hidden) return;

    // Restore original properties
    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    // Re-insert widget at original position
    const targetIndex = widget._originalIndex;
    const insertIndex = Math.min(targetIndex, node.widgets.length);
    node.widgets.splice(insertIndex, 0, widget);

    widget._hidden = false;

    // Show linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => showWidget(node, w));
    }
}

// Refresh node layout after widget visibility change
function refreshNode(node) {
    node.setDirtyCanvas(true, true);
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }

    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);

        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }
    });
}

// Main extension registration
app.registerExtension({
    name: "comfyui.sam3.video_dynamic",

    async nodeCreated(node) {
        if (node.comfyClass === "SAM3VideoSegmentation") {
            const ext = this;
            node._sam3_configured = false;

            // Hook into configure to handle workflow reload
            const origConfigure = node.configure;
            node.configure = function(data) {
                node._sam3_configured = true;
                const result = origConfigure?.apply(this, arguments);
                // Run after configure fully completes and DOM settles
                requestAnimationFrame(() => {
                    ext.setupVideoSegmentation(node);
                });
                return result;
            };

            // Fresh node only (not workflow reload)
            // Double rAF ensures configure has had a chance to run first
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (!node._sam3_configured) {
                        ext.setupVideoSegmentation(node);
                    }
                });
            });
        }
    },

    setupVideoSegmentation(node) {
        log("Setting up SAM3VideoSegmentation node");

        // Find the prompt_mode widget
        const promptModeWidget = node.widgets?.find(w => w.name === "prompt_mode");
        if (!promptModeWidget) {
            log("ERROR: prompt_mode widget not found!");
            return;
        }

        // Find text_prompt widget (this is a widget, not an input slot)
        const textPromptWidget = node.widgets?.find(w => w.name === "text_prompt");

        // Only toggle the text_prompt widget — input slots stay visible always.
        // Splicing node.inputs corrupts connection slot indices on workflow reload.
        const updateVisibility = (mode) => {
            log("Updating visibility for mode:", mode);

            if (textPromptWidget) {
                if (mode === "text") {
                    showWidget(node, textPromptWidget);
                } else {
                    hideWidget(node, textPromptWidget);
                }
            }

            refreshNode(node);
        };

        // Store original callback
        const origCallback = promptModeWidget.callback;

        // Override callback to update visibility when mode changes
        promptModeWidget.callback = function(value) {
            log("prompt_mode changed to:", value);
            const result = origCallback?.apply(this, arguments);
            updateVisibility(value);
            return result;
        };

        // Initialize visibility
        updateVisibility(promptModeWidget.value);
    }
});

log("SAM3 Video dynamic inputs extension loaded");
