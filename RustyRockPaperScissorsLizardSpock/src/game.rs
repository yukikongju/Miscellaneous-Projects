use rand::Rng;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Type {
    ROCK, 
    PAPER, 
    SCISSORS, 
    LIZARD, 
    SPOCK
}

impl Type {
    pub fn get_beats(&self) ->  Vec<Type> {
        match self {
            &Type::ROCK => vec![ Type::SCISSORS, Type::LIZARD ], 
            &Type::PAPER => vec![ Type::ROCK, Type::SPOCK ], 
            &Type::SCISSORS => vec![ Type::PAPER, Type::LIZARD ], 
            &Type::LIZARD => vec![ Type::PAPER, Type::SPOCK ], 
            &Type::SPOCK => vec![ Type::ROCK, Type::SCISSORS ], 
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) ->  fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn generate_random_play() ->  Type {
    match rand::thread_rng().gen_range(0,5) {
        0 => Type::ROCK,
        1 => Type::PAPER,
        2 => Type::SCISSORS,
        3 => Type::LIZARD,
        4 => Type::SPOCK,
        _ => panic!("Error when generating random play")
    }
}

pub fn parse_type(name: String) ->  Option<Type> {
    for typ in vec![ Type::ROCK, Type::PAPER, Type::SCISSORS, Type::LIZARD, Type::SPOCK] {
        if typ.to_string().to_lowercase() == name.to_lowercase() {
            return Some(typ);
        }
    }
    None
}


pub enum Result {
    WIN, 
    LOSE, 
    DRAW
}

pub fn get_player_result(computer_play: &Type, player_play: &Type) -> Result {
    let user_beats = player_play.get_beats();
    let computer_beats = computer_play.get_beats();
    if user_beats.contains(&computer_play) {
        Result::WIN
    } else if computer_beats.contains(&player_play) {
        Result::LOSE
    } else {
        Result::DRAW
    }

}


