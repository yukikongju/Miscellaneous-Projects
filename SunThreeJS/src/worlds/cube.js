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

	// create cube
	const geometry = new THREE.BoxGeometry(1, 1, 1);
	const material = new THREE.MeshBasicMaterial({color: 0xFFFFFF, wireframe: true});
	cube = new THREE.Mesh(geometry, material);

	// add object to scene
	scene.add(cube);
}

function animate() {
	requestAnimationFrame(animate);

	// update object attributes
	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

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
