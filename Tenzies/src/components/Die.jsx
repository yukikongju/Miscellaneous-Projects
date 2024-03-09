export default function Die(props) {
  const styles = {
    backgroundColor: props.isHeld ? "grey" : "default",
  };

  return (
    <button className="die" style={styles} onClick={props.holdDice}>
      {props.value}
    </button>
  );
}
