// Creating Scene
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);


// Update Scene relative to window size
window.addEventListener('resize', function () {
	var width = window.innerWidth;
	var height = window.innerHeight;
	renderer.setSize(width, height);
	camera.aspect = width / height;
	camera.updateProjectionMatrix();
});

// Creating Shape
var geometry = new THREE.BoxGeometry(1, 1, 1);

// Adding material, color, texture
var material = new THREE.MeshBasicMaterial({color: 0xFFFFFF, wireframe: true});
var cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// set camera position
camera.position.z = 3;

// Objects Movements
var update = function () {
	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;
};

// draw scene
var render = function () {
	renderer.render(scene, camera);
};

// run game (update, render, repeat)
var animate = function () {
	requestAnimationFrame(animate);

	update();
	render();
};

animate();
