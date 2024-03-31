import { operators, digits } from "./constants";

export default class Lexer {
  expression: string;

  constructor(expression: string) {
    this.expression = expression;
  }

  tokenize() {
    let tokens = [];
    let i = 0;
    let n = this.expression.length;

    while (i < n) {
      let curr = this.expression[i];

      // check if empty string
      if (curr === " ") {
        continue;
      }

      // check if single operators
      if (operators[curr]) {
        tokens.push(curr);
        i += 1;
        continue;
      }

      // check if trig function or pi
      if (i + 2 < n) {
        let trig = this.expression.slice(i, i + 2);
        if (operators[trig]) {
          tokens.push(trig);
          i += 3;
          continue;
        }
      }
      if (i + 1 < n) {
        let pi = this.expression.slice(i, i + 1);
        if (operators[pi]) {
          tokens.push(pi);
          i += 2;
          continue;
        }
      }

      // check if number
      while (i + 1 < n && digits.includes(this.expression[i + 1])) {
        i += 1;
        curr += this.expression[i + 1];
      }
      tokens.push(curr);
    }

    return tokens;
  }
}
