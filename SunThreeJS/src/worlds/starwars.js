let scene, camera, renderer;
let stars = [];

const MAX = 1000;

function init() {
	// create scene
	scene = new THREE.Scene();

	// create camera
	camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 1000);

	// set camera position
	camera.position.set(0, 0, 500);
	// camera.rotation.x = Math.PI / 2;

	// create renderer
	renderer = new THREE.WebGLRenderer();
	renderer.setPixelRatio(window.devicePixelRatio);
	renderer.setSize(window.innerWidth, window.innerHeight);
	document.body.appendChild(renderer.domElement);

	// add event listener
	window.addEventListener('resize', onWindowResize, false);
}

function animate() {
	requestAnimationFrame(animate);

	// update object attributes
	updateStars();

	// draw environment
	render();
}

function addStars() {
	const geometry = new THREE.SphereGeometry(0.9, 5, 5);
	const material = new THREE.MeshBasicMaterial({color: 0xFFFFFF});

	for (let i = 0, len = 3000; i < len; i++) {
		// create star
		var star = new THREE.Mesh(geometry, material);

		// generate star location
		var x = Math.floor(Math.random() * MAX) - 500;
		var y = Math.floor(Math.random() * MAX) - 500;
		var z = Math.floor(Math.random() * MAX) - 500;
		star.position.set(x, y, z);

		// set star position and adjust scale
		star.scale.x = star.scale.y = 2;

		scene.add(star);
		stars.push(star);
	}
}

function updateStars() {
	for (let i = 0, num = stars.length; i < num; i++) {
		star = stars[i];

		// rotation cinematic effect
		star.rotation.y += 0.01;
		star.rotation.x += 0.07;

		let delta = Math.floor(Math.random() * 0.1);

		star.position.z += i / 1000 + delta;

		// if stars outside the frame, move them back
		if (star.position.z > MAX / 2) {
			star.position.z -= 1000;
			// star.position.z -= 100;
		}
	}
}

function render() {
	renderer.render(scene, camera);
}

function onWindowResize() {
	var width = window.innerWidth;
	var height = window.innerHeight;
	renderer.setSize(width, height);
	camera.aspect = width / height;
	camera.updateProjectionMatrix();
}

init();
addStars();
animate();
