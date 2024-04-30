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

function printEither<E, A>(x: Either<E, A>) {
  if (x._tag == "right") {
    console.log(x.right);
  } else {
    console.log(x.left);
  }
}

// 1. Define Matrix Addition and Multiplication: what is valid/invalid
let a: Scalar = 5;
let x: Matrix = [
  [1, 2],
  [3, 4],
];
// console.log(a);
// console.log(x);

function matrixMultiplication(X: Matrix, Y: Matrix): Either<string, Matrix> {
  // check if size match
  let mx = X.length;
  let nx = X[0].length;
  let my = Y.length;
  let ny = Y[0].length;

  if (nx !== my) {
    return left("undefined");
  }

  // init empty matrix filled with 1s
  let Z: Matrix = [];
  for (let i = 0; i < mx; i++) {
    Z[i] = [];
    for (let j = 0; j < ny; j++) {
      Z[i][j] = 1;
    }
  }

  // perform matrix multiplication
  for (let i = 0; i < mx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nx; k++) {
        Z[i][j] = X[i][k] * Y[k][j];
      }
    }
  }

  return right(Z);
}

function scalarMultiplication(a: Scalar, X: Matrix): Matrix {
  let m = X.length;
  let n = X[0].length;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      X[i][j] *= a;
    }
  }
  return X;
}

function maxtrixAddition(X: Matrix, Y: Matrix): Either<string, Matrix> {
  let mx = X.length;
  let nx = X[0].length;
  let my = Y.length;
  let ny = Y[0].length;

  if (mx !== nx || my !== ny) {
    return left("undefined");
  }

  for (let i = 0; i < mx; i++) {
    for (let j = 0; j < nx; j++) {
      X[i][j] += Y[i][j];
    }
  }
  return right(X);
}

// test functions
let res: Either<string, Matrix> = matrixMultiplication(x, x);
printEither(res);
let res1: Either<string, Matrix> = maxtrixAddition(x, x);
printEither(res1);
let res2: Matrix = scalarMultiplication(a, x);
console.log(res2);

// 2. Define Commutativity Property

// 3. Using Commutativity for different cases with tree parser

// aXbYcdD + dXF
const expression = "aXbYcdD + dXF";

// 4. Try out tests cases and compare compute time
