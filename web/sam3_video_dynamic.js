/**
 * SAM3 Video Segmentation Dynamic Input Management
 *
 * This extension manages the visibility of prompt-mode-specific inputs
 * based on the selected prompt_mode in SAM3VideoSegmentation node.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false;

function log(...args) {
    if (DEBUG) {
        console.log("[SAM3-Video]", ...args);
    }
}

// Canonical order for inputs - positive always before negative
const INPUT_ORDER = [
    "video_frames",
    "positive_points",
    "negative_points",
    "positive_boxes",
    "negative_boxes",
];

// Helper function to hide a widget
function hideWidget(node, widget) {
    if (!widget) return;
    if (widget._hidden) {
        log("Widget already hidden:", widget.name);
        return;
    }

    log("Hiding widget:", widget.name);

    const index = node.widgets.indexOf(widget);
    if (index === -1) {
        log("  ERROR: Widget not found in node.widgets array!");
        return;
    }

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

    log("  Widget removed at index:", index);

    // Hide linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w));
    }
}

// Helper function to show a widget
function showWidget(node, widget) {
    if (!widget) return;
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

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

    log("  Widget restored at index:", insertIndex);

    widget._hidden = false;

    // Show linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => showWidget(node, w));
    }
}

// Helper function to hide an input slot (removes from node.inputs)
function hideInput(node, inputName) {
    if (!node.inputs) return;

    // Check if already in hidden map (already hidden)
    if (node._hiddenInputs && node._hiddenInputs[inputName]) {
        log("  Input already in hidden map:", inputName);
        return;
    }

    const index = node.inputs.findIndex(input => input.name === inputName);
    if (index === -1) {
        log("  Input not found on node:", inputName);
        return;
    }

    const input = node.inputs[index];

    log("Hiding input:", inputName, "at index:", index);

    // Mark as hidden
    input._hidden = true;

    // Store in a hidden inputs map on the node
    if (!node._hiddenInputs) {
        node._hiddenInputs = {};
    }
    node._hiddenInputs[inputName] = input;

    // Disconnect any existing link before removing
    if (input.link !== null) {
        const link = app.graph.links[input.link];
        if (link) {
            const originNode = app.graph.getNodeById(link.origin_id);
            if (originNode) {
                originNode.disconnectOutput(link.origin_slot, node);
            }
        }
    }

    // Remove from inputs array
    node.inputs.splice(index, 1);

    log("  Input removed, remaining inputs:", node.inputs.length);
}

// Helper function to find correct insert position based on canonical order
function findInsertPosition(node, inputName) {
    const targetOrder = INPUT_ORDER.indexOf(inputName);
    if (targetOrder === -1) {
        // Unknown input, add at end
        return node.inputs.length;
    }

    // Find where to insert to maintain canonical order
    for (let i = 0; i < node.inputs.length; i++) {
        const existingOrder = INPUT_ORDER.indexOf(node.inputs[i].name);
        if (existingOrder === -1) continue;
        if (existingOrder > targetOrder) {
            return i;
        }
    }
    return node.inputs.length;
}

// Helper function to show an input slot (restores to node.inputs)
function showInput(node, inputName) {
    // Check if input is already visible
    const existingIndex = node.inputs?.findIndex(input => input.name === inputName);
    if (existingIndex !== -1) {
        log("  Input already visible:", inputName);
        return;
    }

    // Check hidden inputs map
    if (!node._hiddenInputs || !node._hiddenInputs[inputName]) {
        log("  Input not in hidden list:", inputName);
        return;
    }

    const input = node._hiddenInputs[inputName];

    log("Showing input:", inputName);

    // Find the correct position based on canonical order
    const insertIndex = findInsertPosition(node, inputName);

    // Clear hidden flag
    input._hidden = false;

    // Re-insert into inputs array at correct position
    node.inputs.splice(insertIndex, 0, input);

    // Remove from hidden map
    delete node._hiddenInputs[inputName];

    log("  Input restored at index:", insertIndex, "total inputs:", node.inputs.length);
}

// Refresh node layout
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

        requestAnimationFrame(() => {
            if (app.canvas) {
                app.canvas.draw(true, true);
            }
            log("Widget visibility update complete");
        });
    });
}

// Main extension registration
app.registerExtension({
    name: "comfyui.sam3.video_dynamic",

    async nodeCreated(node) {
        if (node.comfyClass === "SAM3VideoSegmentation") {
            // Store reference to extension for configure callback
            const ext = this;

            // Hook into configure to handle workflow reload
            const origConfigure = node.configure;
            node.configure = function(data) {
                const result = origConfigure?.apply(this, arguments);
                // Re-setup after workflow loads
                setTimeout(() => {
                    ext.setupVideoSegmentation(node);
                }, 150);
                return result;
            };

            setTimeout(() => {
                this.setupVideoSegmentation(node);
            }, 100);
        }
    },

    setupVideoSegmentation(node) {
        log("Setting up SAM3VideoSegmentation node");
        log("Initial inputs:", node.inputs?.map(i => i.name));
        log("Initial widgets:", node.widgets?.map(w => w.name));

        // Find the prompt_mode widget
        const promptModeWidget = node.widgets?.find(w => w.name === "prompt_mode");
        if (!promptModeWidget) {
            log("ERROR: prompt_mode widget not found!");
            log("Available widgets:", node.widgets?.map(w => w.name));
            return;
        }

        log("prompt_mode widget found, current value:", promptModeWidget.value);

        // Find text_prompt widget (this is a widget, not an input slot)
        const textPromptWidget = node.widgets?.find(w => w.name === "text_prompt");

        log("Found widgets:", {
            text_prompt: !!textPromptWidget,
        });

        // Ensure _hiddenInputs map exists
        if (!node._hiddenInputs) {
            node._hiddenInputs = {};
        }

        // Pre-populate _hiddenInputs with any mode-specific inputs that exist
        // This handles workflow reload where inputs exist but aren't in the hidden map
        const modeInputs = ["positive_points", "negative_points", "positive_boxes", "negative_boxes"];
        for (const inputName of modeInputs) {
            const existingIndex = node.inputs?.findIndex(input => input.name === inputName);
            if (existingIndex !== -1 && !node._hiddenInputs[inputName]) {
                // Input exists on node but not in hidden map - we'll need to manage it
                log("Pre-registering existing input for management:", inputName);
            }
        }

        // Function to update visibility based on mode
        const updateVisibility = (mode) => {
            log("Updating visibility for mode:", mode);

            // Text mode widget
            if (textPromptWidget) {
                if (mode === "text") {
                    showWidget(node, textPromptWidget);
                } else {
                    hideWidget(node, textPromptWidget);
                }
            }

            // Point mode inputs (these are input SLOTS, not widgets)
            // Box inputs are ALWAYS visible (can be used with any mode for region hints)
            // Point inputs only visible in point mode
            if (mode === "point") {
                // Ensure inputs are shown in correct order: positive first, then negative
                showInput(node, "positive_points");
                showInput(node, "negative_points");
            } else {
                // Text or box mode - hide point inputs
                hideInput(node, "positive_points");
                hideInput(node, "negative_points");
            }

            // Box inputs always visible - can be combined with text or used alone
            showInput(node, "positive_boxes");
            showInput(node, "negative_boxes");

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
        log("Initializing visibility...");
        updateVisibility(promptModeWidget.value);
    }
});

log("SAM3 Video dynamic inputs extension loaded");
