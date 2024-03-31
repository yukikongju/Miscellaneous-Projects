import Lexer from "./lexer";

const expression = "-3 + 34";
let lexer = new Lexer(expression);
let tokens = lexer.tokenize();
console.log(tokens);
