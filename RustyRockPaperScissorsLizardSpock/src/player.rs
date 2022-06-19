use std::fmt;

pub enum Type {
    PLAYER, 
    COMPUTER
}

#[derive(Debug, PartialEq)]
pub struct Player {
    pub name: String,
    pub score: i32,
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:0} : {:1}", self.name, self.score)
    }
}

impl Player {
    pub fn new(name: &str) -> Player {
        Player {
            name: String::from(name),
            score: 0,
        }
    }
    pub fn increment_score(&mut self) {
        self.score += 1;
    }
}

