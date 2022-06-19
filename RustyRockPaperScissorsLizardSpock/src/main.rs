extern crate rusty_rock_paper_scissors_lizard_spock;

// use std::io::{stdin, stdout, Write};
use std::cmp::{Ordering};
use rusty_rock_paper_scissors_lizard_spock::game::{self, Result};
use rusty_rock_paper_scissors_lizard_spock::player::{self, Player};
use rusty_rock_paper_scissors_lizard_spock::utils::{read_line};

fn main() {
    // initialize game variables
    let mut player: Player = Player::new("player");
    let mut computer: Player = Player::new("computer");
    let num_games: i32 = 5;

    // play n games
    for x in 0..num_games {
        println!("------------------- [ ROUND {} ] -------------", x + 1);

        // get user and computer input
        let user_play: game::Type = parse_play();
        let computer_play: game::Type = game::generate_random_play();

        // check result: win, lose, draw
        let result = game::get_player_result(&computer_play, &user_play);
        println!("You played {}. Computer played {}", &user_play, &computer_play);
        match result {
            Result::DRAW => {
                println!("It's a draw!");
                continue
            }, 
            Result::WIN => {
                println!("You won!");
                player.increment_score();
            }, 
            Result::LOSE => {
                println!("You lost :(");
                computer.increment_score();
            }
        }

        // print current score
        println!("Player: {} Computer: {} ", &player.score, &computer.score);
        
    }

    // print outcomes
    match &player.score.cmp(&computer.score) {
        Ordering::Less => { println!("You lost:(") }, 
        Ordering::Greater => { println!("You won!") }, 
        Ordering::Equal => { println!("It's a draw") }, 
    }

}

fn parse_play() -> game::Type {
    // println!("What to play? (r:rock, p:paper, s:scissors, l:lizard, v:spock)");
    println!("What to play? (rock, paper, scissors, lizard, spock)");
    let input = read_line();
    match game::parse_type(input) {
        Some(x) => x,
        None => {
            print!("Invalid value");
            parse_play()
        }
    }
}

