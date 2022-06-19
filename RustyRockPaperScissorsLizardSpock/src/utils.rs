use std::io::{self};

pub fn read_line() ->  String {
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Unable to read line");
    input.pop(); // remove \n
    input
}

