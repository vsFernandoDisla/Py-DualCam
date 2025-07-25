function Refresh() {
  fetch("http://127.0.0.1:8000/latest_frames")
    .then((response) => response.json())
    .then((data) => {
      const container = document.getElementById("frames");

      for (const [camName, base64Image] of Object.entries(data)) {
        const title = document.createElement("h2");
        title.textContent = `Cámara: ${camName}`;
        container.appendChild(title);

        const img = document.createElement("img");
        img.src = base64Image;
        img.style.width = "400px";
        img.style.margin = "10px";
        container.appendChild(img);
      }
    })
    .catch((err) => {
      document.getElementById("frames").textContent = "Error cargando frames: " + err;
    });
}

function Record() {
  fetch("http://127.0.0.1:8000/save_frames", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
    })
    .catch((err) => {
      document.getElementById("frames").textContent = "Error grabando frames: " + err;
    });
}
