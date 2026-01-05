/**
 * SAM3 Multi-Object Point Collector
 * Uses plain HTML5 Canvas with support for multiple object ID grouping
 * Version: 2025-01-20-v9-MULTI-OBJECT
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

console.log("[SAM3] ===== VERSION 9 - MULTI-OBJECT SUPPORT =====");

// Color palette for different objects (matching SAM3VideoSegmenter.COLORS)
const OBJECT_COLORS = [
    '#0080FF',  // Blue (obj 1)
    '#FF4D4D',  // Red (obj 2)
    '#4DFF4D',  // Green (obj 3)
    '#FFFF00',  // Yellow (obj 4)
    '#FF00FF',  // Magenta (obj 5)
    '#00FFFF',  // Cyan (obj 6)
    '#FF8000',  // Orange (obj 7)
    '#8000FF',  // Purple (obj 8)
    '#80FF00',  // Lime (obj 9+)
    '#FF0080',  // Pink
];

// Negative point color (consistent across all objects)
const NEGATIVE_COLOR = '#FF0000';

// Helper function to properly hide widgets (enhanced for complete hiding)
function hideWidgetForGood(node, widget, suffix = '') {
    if (!widget) return;

    // Save original properties
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;

    // Multiple hiding approaches to ensure widget is fully hidden
    widget.computeSize = () => [0, -4];  // -4 compensates for litegraph's automatic widget gap
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;  // Mark as hidden

    // IMPORTANT: Keep serialization enabled so values are sent to backend
    // (We just hide it visually, but it still needs to send data)

    // Make the widget completely invisible in the DOM if it has element
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }

    // Handle linked widgets recursively
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name);
        }
    }
}

app.registerExtension({
    name: "Comfy.SAM3.SimplePointCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[SAM3] beforeRegisterNodeDef called for:", nodeData.name);

        if (nodeData.name === "SAM3PointCollector") {
            console.log("[SAM3] Registering SAM3PointCollector node with multi-object support");
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                console.log("[SAM3] onNodeCreated called for SAM3PointCollector");

                // Call original onNodeCreated FIRST to create all widgets
                const result = onNodeCreated?.apply(this, arguments);

                console.log("[SAM3] Widgets after creation:", this.widgets?.map(w => w.name));

                console.log("[SAM3] Creating canvas container with multi-object support");
                // Create canvas container - dynamically sized based on node height
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; margin: 0; padding: 0; display: flex; flex-direction: column;";

                // Create top info/button bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: relative; padding: 5px 10px; z-index: 10; display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.5); flex-shrink: 0;";
                container.appendChild(infoBar);

                // Left side: Current object indicator and New Object button
                const leftControls = document.createElement("div");
                leftControls.style.cssText = "display: flex; align-items: center; gap: 8px;";
                infoBar.appendChild(leftControls);

                // Current object indicator
                const objectIndicator = document.createElement("div");
                objectIndicator.style.cssText = "padding: 5px 10px; background: " + OBJECT_COLORS[0] + "; color: #000; border-radius: 3px; font-size: 12px; font-family: monospace; font-weight: bold;";
                objectIndicator.textContent = "Object 1";
                leftControls.appendChild(objectIndicator);

                // New Object button
                const newObjectButton = document.createElement("button");
                newObjectButton.textContent = "+ New Object (N)";
                newObjectButton.style.cssText = "padding: 5px 10px; background: #4a4; color: #fff; border: 1px solid #2a2; border-radius: 3px; cursor: pointer; font-size: 11px; font-weight: bold;";
                newObjectButton.onmouseover = () => newObjectButton.style.background = "#5b5";
                newObjectButton.onmouseout = () => newObjectButton.style.background = "#4a4";
                leftControls.appendChild(newObjectButton);

                // Right side: Points counter and clear button
                const rightControls = document.createElement("div");
                rightControls.style.cssText = "display: flex; align-items: center; gap: 8px;";
                infoBar.appendChild(rightControls);

                // Points counter
                const pointsCounter = document.createElement("div");
                pointsCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 11px; font-family: monospace;";
                pointsCounter.textContent = "Obj 1: 0 pts";
                rightControls.appendChild(pointsCounter);

                // Clear button
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear All";
                clearButton.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 11px; font-weight: bold;";
                clearButton.onmouseover = () => clearButton.style.background = "#e55";
                clearButton.onmouseout = () => clearButton.style.background = "#d44";
                rightControls.appendChild(clearButton);

                // Create object list panel
                const objectListPanel = document.createElement("div");
                objectListPanel.style.cssText = "position: relative; padding: 5px 10px; background: rgba(0,0,0,0.3); display: flex; flex-wrap: wrap; gap: 5px; flex-shrink: 0; min-height: 30px; align-items: center;";
                container.appendChild(objectListPanel);

                // Create canvas wrapper
                const canvasWrapper = document.createElement("div");
                canvasWrapper.style.cssText = "flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; min-height: 200px;";
                container.appendChild(canvasWrapper);

                // Create canvas for image and points
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair;";
                canvasWrapper.appendChild(canvas);

                const ctx = canvas.getContext("2d");
                console.log("[SAM3] Canvas created with multi-object support:", canvas);

                // Store state - NEW: Multi-object structure
                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    image: null,
                    // Multi-object data structure
                    objects: [
                        { obj_id: 1, positivePoints: [], negativePoints: [] }
                    ],
                    currentObjectIndex: 0,  // Index into objects array
                    hoveredPoint: null,
                    pointsCounter: pointsCounter,
                    objectIndicator: objectIndicator,
                    objectListPanel: objectListPanel
                };

                // Add as DOM widget
                console.log("[SAM3] Adding DOM widget via addDOMWidget");
                const widget = this.addDOMWidget("canvas", "customCanvas", container);
                console.log("[SAM3] addDOMWidget returned:", widget);

                // Store widget reference for updates
                this.canvasWidget.domWidget = widget;

                // Make widget dynamically sized - override computeSize
                widget.computeSize = (width) => {
                    // Widget height = node height - title bar/padding (approx 80px)
                    const nodeHeight = this.size ? this.size[1] : 540;
                    const widgetHeight = Math.max(250, nodeHeight - 80);
                    return [width, widgetHeight];
                };

                // New Object button handler
                newObjectButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.addNewObject();
                });

                // Clear button handler
                clearButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log("[SAM3] Clearing all objects and points");
                    this.canvasWidget.objects = [
                        { obj_id: 1, positivePoints: [], negativePoints: [] }
                    ];
                    this.canvasWidget.currentObjectIndex = 0;
                    this.updatePoints();
                    this.updateObjectList();
                    this.redrawCanvas();
                });

                // Hide the string storage widgets - multiple approaches
                console.log("[SAM3] Attempting to hide widgets...");
                console.log("[SAM3] Widgets before hiding:", this.widgets.map(w => w.name));

                const coordsWidget = this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this.widgets.find(w => w.name === "points_store");

                console.log("[SAM3] Found widgets to hide:", { coordsWidget, negCoordsWidget, storeWidget });

                // Initialize default values BEFORE hiding
                if (coordsWidget) {
                    coordsWidget.value = coordsWidget.value || "[]";
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = negCoordsWidget.value || "[]";
                }
                if (storeWidget) {
                    storeWidget.value = storeWidget.value || "{}";
                }

                // Store references before hiding
                this._hiddenWidgets = {
                    coordinates: coordsWidget,
                    neg_coordinates: negCoordsWidget,
                    points_store: storeWidget
                };

                // Apply hiding
                if (coordsWidget) {
                    hideWidgetForGood(this, coordsWidget);
                    console.log("[SAM3] coordinates - type:", coordsWidget.type, "hidden:", coordsWidget.hidden, "value:", coordsWidget.value);
                }
                if (negCoordsWidget) {
                    hideWidgetForGood(this, negCoordsWidget);
                    console.log("[SAM3] neg_coordinates - type:", negCoordsWidget.type, "hidden:", negCoordsWidget.hidden, "value:", negCoordsWidget.value);
                }
                if (storeWidget) {
                    hideWidgetForGood(this, storeWidget);
                    console.log("[SAM3] points_store - type:", storeWidget.type, "hidden:", storeWidget.hidden, "value:", storeWidget.value);
                }

                // CRITICAL FIX: Override onDrawForeground to skip rendering hidden widgets
                const originalDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    // Temporarily hide converted widgets from rendering
                    const hiddenWidgets = this.widgets.filter(w => w.type?.includes("converted-widget"));
                    const originalTypes = hiddenWidgets.map(w => w.type);

                    // Temporarily set to null to prevent rendering
                    hiddenWidgets.forEach(w => w.type = null);

                    // Call original draw
                    if (originalDrawForeground) {
                        originalDrawForeground.apply(this, arguments);
                    }

                    // Restore types
                    hiddenWidgets.forEach((w, i) => w.type = originalTypes[i]);

                    // Update container height based on current node size
                    const containerHeight = Math.max(250, this.size[1] - 80);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                };

                console.log("[SAM3] Widgets after hiding:", this.widgets.map(w => `${w.name}(${w.type})`));
                console.log("[SAM3] All widgets processing complete");

                // Mouse event handlers
                // Left-click = positive point
                // Shift+click or Ctrl+click = negative point
                // Alt+click on existing point = delete
                canvas.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    console.log(`[SAM3] Click at canvas coords: (${x.toFixed(1)}, ${y.toFixed(1)}), canvas size: ${canvas.width}x${canvas.height}`);

                    const currentObj = this.canvasWidget.objects[this.canvasWidget.currentObjectIndex];

                    // Check if clicking existing point to delete (Alt+click)
                    const clickedPoint = this.findPointAt(x, y);
                    if (clickedPoint && e.altKey) {
                        // Alt+click on existing point = delete
                        const obj = this.canvasWidget.objects[clickedPoint.objectIndex];
                        if (clickedPoint.type === 'positive') {
                            obj.positivePoints = obj.positivePoints.filter(p => p !== clickedPoint.point);
                        } else {
                            obj.negativePoints = obj.negativePoints.filter(p => p !== clickedPoint.point);
                        }
                        console.log(`[SAM3] Deleted ${clickedPoint.type} point from object ${clickedPoint.objectIndex + 1}`);
                    } else {
                        // Add new point to current object
                        if (e.shiftKey || e.ctrlKey) {
                            // Shift+click or Ctrl+click = Negative point
                            currentObj.negativePoints.push({x, y});
                            console.log(`[SAM3] Added negative point to object ${this.canvasWidget.currentObjectIndex + 1}`);
                        } else {
                            // Left-click = Positive point
                            currentObj.positivePoints.push({x, y});
                            console.log(`[SAM3] Added positive point to object ${this.canvasWidget.currentObjectIndex + 1}`);
                        }
                    }

                    this.updatePoints();
                    this.updateObjectList();
                    this.redrawCanvas();
                });

                // Prevent context menu on canvas (just block it, don't use for points)
                canvas.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });

                canvas.addEventListener("mousemove", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                    const hovered = this.findPointAt(x, y);
                    if (hovered !== this.canvasWidget.hoveredPoint) {
                        this.canvasWidget.hoveredPoint = hovered;
                        this.redrawCanvas();
                    }
                });

                // Keyboard shortcuts - store handler for cleanup on node removal
                this._keydownHandler = (e) => {
                    // Only handle if this node is focused/selected
                    if (!this.is_selected) return;

                    if (e.key === 'n' || e.key === 'N' || e.key === 'Tab') {
                        e.preventDefault();
                        this.addNewObject();
                    } else if (e.key >= '1' && e.key <= '9') {
                        const objIndex = parseInt(e.key) - 1;
                        if (objIndex < this.canvasWidget.objects.length) {
                            this.switchToObject(objIndex);
                        }
                    } else if (e.key === 'Delete' || e.key === 'Backspace') {
                        // Delete last point from current object
                        const currentObj = this.canvasWidget.objects[this.canvasWidget.currentObjectIndex];
                        if (currentObj.positivePoints.length > 0) {
                            currentObj.positivePoints.pop();
                        } else if (currentObj.negativePoints.length > 0) {
                            currentObj.negativePoints.pop();
                        }
                        this.updatePoints();
                        this.updateObjectList();
                        this.redrawCanvas();
                    }
                };
                document.addEventListener("keydown", this._keydownHandler);

                // Handle image input changes
                this.onExecuted = (message) => {
                    console.log("[SAM3] onExecuted called with message:", message);
                    if (message.bg_image && message.bg_image[0]) {
                        const img = new Image();
                        img.onload = () => {
                            console.log(`[SAM3] Image loaded: ${img.width}x${img.height}`);
                            this.canvasWidget.image = img;
                            canvas.width = img.width;
                            canvas.height = img.height;
                            console.log(`[SAM3] Canvas resized to: ${canvas.width}x${canvas.height}`);
                            this.redrawCanvas();
                        };
                        img.src = "data:image/jpeg;base64," + message.bg_image[0];
                    }
                };

                // Update container height dynamically when node size changes
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.apply(this, arguments);
                    }
                    // Update container to match widget size
                    const containerHeight = Math.max(250, size[1] - 80);
                    container.style.height = containerHeight + "px";
                    console.log(`[SAM3] Node resized to: ${size[0]}x${size[1]}, container height: ${containerHeight}px`);
                };

                // Also update on draw to handle any size changes
                const originalOnDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    if (originalOnDrawForeground) {
                        originalOnDrawForeground.apply(this, arguments);
                    }
                    // Update container height based on current node size
                    const containerHeight = Math.max(250, this.size[1] - 80);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                };

                // Draw initial state
                console.log("[SAM3] Drawing initial state");
                this.updateObjectList();
                this.redrawCanvas();

                // Set initial node size
                const nodeWidth = Math.max(450, this.size[0] || 450);
                const nodeHeight = 540; // Initial height: canvas + controls
                this.setSize([nodeWidth, nodeHeight]);

                // Set initial container height
                container.style.height = "460px";

                console.log("[SAM3] Node size set to:", [nodeWidth, nodeHeight]);
                console.log("[SAM3] onNodeCreated complete with multi-object support");
                return result;
            };

            // Cleanup keyboard listener when node is removed (prevents memory leak)
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this._keydownHandler) {
                    document.removeEventListener("keydown", this._keydownHandler);
                    this._keydownHandler = null;
                    console.log("[SAM3] Cleaned up keyboard listener on node removal");
                }
                if (originalOnRemoved) {
                    originalOnRemoved.apply(this, arguments);
                }
            };

            // Helper: Add new object
            nodeType.prototype.addNewObject = function() {
                const newObjId = this.canvasWidget.objects.length + 1;
                this.canvasWidget.objects.push({
                    obj_id: newObjId,
                    positivePoints: [],
                    negativePoints: []
                });
                this.canvasWidget.currentObjectIndex = this.canvasWidget.objects.length - 1;
                console.log(`[SAM3] Added new object ${newObjId}, now have ${this.canvasWidget.objects.length} objects`);
                this.updateObjectIndicator();
                this.updateObjectList();
                this.updatePoints();
            };

            // Helper: Switch to specific object
            nodeType.prototype.switchToObject = function(index) {
                if (index >= 0 && index < this.canvasWidget.objects.length) {
                    this.canvasWidget.currentObjectIndex = index;
                    console.log(`[SAM3] Switched to object ${index + 1}`);
                    this.updateObjectIndicator();
                    this.updateObjectList();
                    this.redrawCanvas();
                }
            };

            // Helper: Update object indicator
            nodeType.prototype.updateObjectIndicator = function() {
                const currentIdx = this.canvasWidget.currentObjectIndex;
                const color = OBJECT_COLORS[currentIdx % OBJECT_COLORS.length];
                this.canvasWidget.objectIndicator.style.background = color;
                this.canvasWidget.objectIndicator.textContent = `Object ${currentIdx + 1}`;

                // Update points counter for current object
                const currentObj = this.canvasWidget.objects[currentIdx];
                const posCount = currentObj.positivePoints.length;
                const negCount = currentObj.negativePoints.length;
                this.canvasWidget.pointsCounter.textContent = `Obj ${currentIdx + 1}: ${posCount} pos, ${negCount} neg`;
            };

            // Helper: Update object list panel
            nodeType.prototype.updateObjectList = function() {
                const panel = this.canvasWidget.objectListPanel;
                panel.innerHTML = '';

                this.canvasWidget.objects.forEach((obj, idx) => {
                    const totalPoints = obj.positivePoints.length + obj.negativePoints.length;
                    const isActive = idx === this.canvasWidget.currentObjectIndex;
                    const color = OBJECT_COLORS[idx % OBJECT_COLORS.length];

                    const objButton = document.createElement("button");
                    objButton.style.cssText = `
                        padding: 3px 8px;
                        background: ${isActive ? color : '#444'};
                        color: ${isActive ? '#000' : '#fff'};
                        border: 2px solid ${color};
                        border-radius: 3px;
                        cursor: pointer;
                        font-size: 10px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    `;

                    // Color dot
                    const colorDot = document.createElement("span");
                    colorDot.style.cssText = `width: 8px; height: 8px; background: ${color}; border-radius: 50%; display: inline-block;`;
                    objButton.appendChild(colorDot);

                    // Text
                    const text = document.createElement("span");
                    text.textContent = `${idx + 1}: ${totalPoints}pts`;
                    objButton.appendChild(text);

                    // Delete button (only if more than 1 object)
                    if (this.canvasWidget.objects.length > 1) {
                        const deleteBtn = document.createElement("span");
                        deleteBtn.textContent = "×";
                        deleteBtn.style.cssText = "margin-left: 4px; color: #f66; font-weight: bold; cursor: pointer;";
                        deleteBtn.onclick = (e) => {
                            e.stopPropagation();
                            this.deleteObject(idx);
                        };
                        objButton.appendChild(deleteBtn);
                    }

                    objButton.onclick = () => this.switchToObject(idx);
                    panel.appendChild(objButton);
                });

                this.updateObjectIndicator();
            };

            // Helper: Delete object
            nodeType.prototype.deleteObject = function(index) {
                if (this.canvasWidget.objects.length <= 1) return;

                this.canvasWidget.objects.splice(index, 1);

                // Renumber obj_ids
                this.canvasWidget.objects.forEach((obj, idx) => {
                    obj.obj_id = idx + 1;
                });

                // Adjust current index if needed
                if (this.canvasWidget.currentObjectIndex >= this.canvasWidget.objects.length) {
                    this.canvasWidget.currentObjectIndex = this.canvasWidget.objects.length - 1;
                }

                console.log(`[SAM3] Deleted object at index ${index}, now have ${this.canvasWidget.objects.length} objects`);
                this.updateObjectList();
                this.updatePoints();
                this.redrawCanvas();
            };

            // Helper: Find point at coordinates (searches all objects)
            nodeType.prototype.findPointAt = function(x, y) {
                const threshold = 10;

                for (let objIdx = 0; objIdx < this.canvasWidget.objects.length; objIdx++) {
                    const obj = this.canvasWidget.objects[objIdx];

                    for (const point of obj.positivePoints) {
                        if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                            return {type: 'positive', point, objectIndex: objIdx};
                        }
                    }

                    for (const point of obj.negativePoints) {
                        if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                            return {type: 'negative', point, objectIndex: objIdx};
                        }
                    }
                }

                return null;
            };

            // Helper: Update widget values - NEW: Multi-object format
            nodeType.prototype.updatePoints = function() {
                // Use stored hidden widget references
                const coordsWidget = this._hiddenWidgets?.coordinates || this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this._hiddenWidgets?.neg_coordinates || this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this._hiddenWidgets?.points_store || this.widgets.find(w => w.name === "points_store");

                // Build multi-object output format
                const multiObjectData = {
                    objects: this.canvasWidget.objects.map(obj => ({
                        obj_id: obj.obj_id,
                        positive_points: obj.positivePoints.map(p => [p.x, p.y]),
                        negative_points: obj.negativePoints.map(p => [p.x, p.y])
                    }))
                };

                // For backward compatibility, also output flattened single-object format
                // (combines all positive/negative points from all objects)
                const allPositive = [];
                const allNegative = [];
                for (const obj of this.canvasWidget.objects) {
                    allPositive.push(...obj.positivePoints);
                    allNegative.push(...obj.negativePoints);
                }

                if (coordsWidget) {
                    coordsWidget.value = JSON.stringify(allPositive);
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = JSON.stringify(allNegative);
                }
                if (storeWidget) {
                    // Store BOTH formats: multi-object (primary) and legacy (for backward compat)
                    storeWidget.value = JSON.stringify({
                        // New multi-object format
                        objects: multiObjectData.objects,
                        // Legacy format (for backward compatibility)
                        positive: allPositive,
                        negative: allNegative
                    });
                }

                // Update points counter display
                const currentIdx = this.canvasWidget.currentObjectIndex;
                const currentObj = this.canvasWidget.objects[currentIdx];
                const posCount = currentObj.positivePoints.length;
                const negCount = currentObj.negativePoints.length;
                this.canvasWidget.pointsCounter.textContent = `Obj ${currentIdx + 1}: ${posCount} pos, ${negCount} neg`;
            };

            // Helper: Redraw canvas
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, objects, currentObjectIndex, hoveredPoint} = this.canvasWidget;

                // Clear
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw image if available
                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    // Placeholder
                    ctx.fillStyle = "#333";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Click to add points", canvas.width / 2, canvas.height / 2 - 25);
                    ctx.fillText("Left-click: Positive point", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Shift/Right-click: Negative point", canvas.width / 2, canvas.height / 2 + 25);
                    ctx.fillText("Press N or Tab: New object", canvas.width / 2, canvas.height / 2 + 50);
                    ctx.fillText("Press 1-9: Switch object", canvas.width / 2, canvas.height / 2 + 75);
                }

                // Draw canvas dimensions overlay (helpful for debugging)
                if (image) {
                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    ctx.fillRect(5, canvas.height - 25, 150, 20);
                    ctx.fillStyle = "#0f0";
                    ctx.font = "12px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
                }

                // Draw all objects' points
                for (let objIdx = 0; objIdx < objects.length; objIdx++) {
                    const obj = objects[objIdx];
                    const color = OBJECT_COLORS[objIdx % OBJECT_COLORS.length];
                    const isCurrentObject = objIdx === currentObjectIndex;

                    // Draw positive points in object color
                    ctx.strokeStyle = color;
                    ctx.fillStyle = color;
                    for (const point of obj.positivePoints) {
                        const isHovered = hoveredPoint?.point === point;
                        const baseRadius = isCurrentObject ? 7 : 5;
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, isHovered ? baseRadius + 2 : baseRadius, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.lineWidth = isCurrentObject ? 2 : 1;
                        ctx.stroke();

                        // Draw object ID label on the point
                        ctx.fillStyle = "#000";
                        ctx.font = "bold 10px sans-serif";
                        ctx.textAlign = "center";
                        ctx.fillText(String(objIdx + 1), point.x, point.y + 4);
                        ctx.fillStyle = color;
                    }

                    // Draw negative points in red with object indicator
                    ctx.strokeStyle = NEGATIVE_COLOR;
                    ctx.fillStyle = NEGATIVE_COLOR;
                    for (const point of obj.negativePoints) {
                        const isHovered = hoveredPoint?.point === point;
                        const baseRadius = isCurrentObject ? 7 : 5;
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, isHovered ? baseRadius + 2 : baseRadius, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.lineWidth = isCurrentObject ? 2 : 1;

                        // Draw border in object color to show which object it belongs to
                        ctx.strokeStyle = color;
                        ctx.stroke();

                        // Draw X mark for negative
                        ctx.strokeStyle = "#000";
                        ctx.lineWidth = 2;
                        const r = 3;
                        ctx.beginPath();
                        ctx.moveTo(point.x - r, point.y - r);
                        ctx.lineTo(point.x + r, point.y + r);
                        ctx.moveTo(point.x + r, point.y - r);
                        ctx.lineTo(point.x - r, point.y + r);
                        ctx.stroke();
                    }
                }
            };
        }
    }
});
