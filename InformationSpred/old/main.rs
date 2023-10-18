use rand::distribution::{Distribution, Normal};
use rand::Rng;

struct Person {
    friends: Vec<usize>,
    has_info: bool,
    share_probability: f64,
}

pub fn main() {
    let num_people = 100;
    let mean_friends = 10.0;
    let std_deviation_friends = 3.0;
    let mean_share_prob = 75.0;
    let std_deviation_share_prob = 20.0;
    let info_prob = 0.5;

    let mut rng = rand::thread_rng();
    let normal_distribution_friends = Normal::new(mean_friends, std_deviation_friends);
    let normal_distribution_share_prob = Normal::new(mean_share_prob, std_deviation_share_prob);

    // initializaing the network
    let mut people: Vec<Person> Vec::with_capacity(num_people);
    for _ in 0..num_people {
        let num_friends = rng.sample(normal_distribution_friends) as usize;
        let friends = (0..num_friends).map(|_| rng.gen_range(0, num_people)).collect();
        let share_prob = rng.sample(normal_distribution_share_prob) as usize;
        let has_info = is_above_threshold(info_prob);
        let person = Person {
            friends,
            has_info,
            share_prob
        };
        people.push(person);
    }

    // run the simulation
    let mut iteration = 0;
    let mut info_spread_count = 0;

    while info_spread_count < num_people {
        for (i, person) in people.iter_mut().enumerate() {
            // check if person share info with friends
            for &friend_id in &person.friends {
                let friend = &mut people[friend_id];
                if person.has_info && !friend.has_info {
                    friend.has_info = true;
                    info_spread_count += 1;
                }
            }

        }

        iteration += 1;

    }

    println!("Number of iterations required: {} iterations", iteration);




}


pub fn is_above_threshold(threshold: f64) -> bool {
    let random_number = rand::thread_rng().gen::<f64>();
    random_number > threshold
}


