// 0. defining types: Matrix, Scalar, Either

type Matrix = number[][];
type Scalar = number;

type Either<E, A> = Left<E> | Right<A>;
interface Left<E> {
  _tag: "left";
  left: E;
}
interface Right<A> {
  _tag: "right";
  right: A;
}
const left = <E, A = never>(e: E): Either<E, A> => ({
  _tag: "left",
  left: e,
});
const right = <A, E = never>(a: A): Either<E, A> => ({
  _tag: "right",
  right: a,
});

// 1. Define Matrix Addition and Multiplication: what is valid/invalid

function matrixMultiplication(X: Matrix, Y: Matrix): Either<string, Matrix> {
  // check if size match
  let mx = X.length;
  let nx = X[0].length;
  let my = Y.length;
  let ny = Y[0].length;
  if (nx == my) {
    return left("undefined");
  }

  // perform matrix multiplication
  const Z: Matrix = Array.from({ length: mx }, () => Array(ny).fill(1));
  for (let i = 0; i < mx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < mx; k++) {
        Z[i][j] = X[i][k] * Y[k][j];
      }
    }
  }

  return right(Z);
}
const x: Matrix = [
  [1, 2],
  [3, 4],
];
console.log(x);
// const res: Either<string, Matrix> = matrixMultiplication(x, x);
// if (res._tag == "right") {
//         console.log('a');
// } else {
//   print(res.left);
// }

// 2. Define Commutativity Property

// 3. Using Commutativity for different cases

// 4. Try out tests cases and compare compute time
