let scene, camera, renderer, cube;

function init() {
	// create scene
	scene = new THREE.Scene();

	// create camera
	camera = new THREE.PerspectiveCamera(
		75,
		window.innerWidth / window.innerHeight,
		0.1,
		1000
	);

	// set camera position
	camera.position.set(0, 0, 5);

	// create renderer
	renderer = new THREE.WebGLRenderer();
	renderer.setSize(window.innerWidth, window.innerHeight);
	document.body.appendChild(renderer.domElement);

	// TODO: create and add objets to scene
}

function animate() {
	requestAnimationFrame(animate);

	// TODO: update object attributes

	// draw environment
	renderer.render(scene, camera);
}

function onWindowResize() {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	rendere.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onWindowResize, false);

init();
animate();
