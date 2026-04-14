const scenes = {
    start: {
        bg: "https://picsum.photos/800/600",
        hotspots: [
            { x: 40, y: 40, w: 20, h: 20, target: "room" },
        ],
    },

    room: {
        bg: "https://picsum.photos/800/601",
        hotspots: [
            { x: 10, y: 10, w: 20, h: 20, target: "start" },
        ],
    },
};

const bg = document.getElementById("bg");
const layer = document.getElementById("hotspots");

function loadScene(id) {
    const scene = scenes[id];

    bg.src = scene.bg;
    layer.innerHTML = "";

    scene.hotspots.forEach((h) => {
        const el = document.createElement("div");
        el.className = "hotspot";

        el.style.left = h.x + "%";
        el.style.top = h.y + "%";
        el.style.width = h.w + "%";
        el.style.height = h.h + "%";

        el.onclick = () => loadScene(h.target);

        layer.appendChild(el);
    });
}

// start game
loadScene("start");
