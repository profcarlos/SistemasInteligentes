//-----------------------------------------------------------------------------------------------------
// IFG - Campus Goiânia
// Curso: Engenharia Elétrica. 
// Disciplina: Sistemas Inteligentes
// Professor: Carlos Roberto da Silveira Junior
// Objetivo: O Robô faz uso da lógica fuzzy para controle de velocidade tendo como entrada a distância.
// Utilizou-se a biblioteca eFLL para lógica Fuzzy, disponível em https://github.com/zerokol/eFLL.
//-------------------------------------------------------------------------------------------------------

#include <Fuzzy.h>
#include <FuzzyComposition.h>
#include <FuzzyInput.h>
#include <FuzzyIO.h>
#include <FuzzyOutput.h>
#include <FuzzyRule.h>
#include <FuzzyRuleAntecedent.h>
#include <FuzzyRuleConsequent.h>
#include <FuzzySet.h>

// Definição dos pinos 

  int pinL1D = 2;
  int pinL1A = 19;
  int pinL2D = 3;
  int pinL2A = 18;
  int pinUS = 5;
  int pinEB = 6;
  int pinI4 = 7;
  int pinI3 = 8;
  int pinEA = 11;
  int pinI2 = 12;
  int pinI1 = 13;
  int pinTDIR = 14;
  int pinTESQ = 15;
  int pinAPITO = 17;

// Constante de velocidade divisor (1 - 10)
  int constVel = 1;
  
// Definicao de variaveis
  int sensLuz1 = 0;
  int sensLuz2 = 0;
  int valLuz1  = 0;
  int distancia =0;
  int velocidade =0;
  
  
// Step 1 -  Instantiating an object library
  Fuzzy* fuzzy = new Fuzzy();
  
void setup()
{
  pinMode(pinL1D, OUTPUT);
  pinMode(pinL2D, OUTPUT);
  pinMode(pinEA , OUTPUT);
  pinMode(pinEB , OUTPUT);
  pinMode(pinI1 , OUTPUT);
  pinMode(pinI2 , OUTPUT);
  pinMode(pinI3 , OUTPUT);
  pinMode(pinI4 , OUTPUT);
  pinMode(pinTDIR,  OUTPUT);
  pinMode(pinTESQ,  OUTPUT);
  pinMode(pinAPITO, OUTPUT);
  while(!digitalRead(pinTDIR));
  ligaApito();
  delay(1000);
  desligaApito();
  delay(1000);
  Serial.begin(9600);
  Serial.println('Iniciando processo');

  // Step 2 - Creating a FuzzyInput distance
  FuzzyInput* distance = new FuzzyInput(1);// With its ID in param

  // Creating the FuzzySet to compond FuzzyInput distance
  FuzzySet* small = new FuzzySet(0, 10, 10, 40); // Small distance
  distance->addFuzzySet(small); // Add FuzzySet small to distance
  FuzzySet* safe = new FuzzySet(20, 60, 60, 80); // Safe distance
  distance->addFuzzySet(safe); // Add FuzzySet safe to distance
  FuzzySet* big = new FuzzySet(60, 80, 80, 120); // Big distance
  distance->addFuzzySet(big); // Add FuzzySet big to distance

  fuzzy->addFuzzyInput(distance); // Add FuzzyInput to Fuzzy object

  // Passo 3 - Creating FuzzyOutput velocity
  FuzzyOutput* velocity = new FuzzyOutput(1);// With its ID in param

  // Creating FuzzySet to compond FuzzyOutput velocity
  FuzzySet* slow = new FuzzySet(0, 40, 40, 60); // Slow velocity
  velocity->addFuzzySet(slow); // Add FuzzySet slow to velocity
  FuzzySet* average = new FuzzySet(40, 90, 90, 120); // Average velocity
  velocity->addFuzzySet(average); // Add FuzzySet average to velocity
  FuzzySet* fast = new FuzzySet(90, 120, 120, 160); // Fast velocity
  velocity->addFuzzySet(fast); // Add FuzzySet fast to velocity

  fuzzy->addFuzzyOutput(velocity); // Add FuzzyOutput to Fuzzy object

  //Passo 4 - Assembly the Fuzzy rules
  // FuzzyRule "IF distance = samll THEN velocity = slow"
  FuzzyRuleAntecedent* ifDistanceSmall = new FuzzyRuleAntecedent(); // Instantiating an Antecedent to expression
  ifDistanceSmall->joinSingle(small); // Adding corresponding FuzzySet to Antecedent object
  FuzzyRuleConsequent* thenVelocitySlow = new FuzzyRuleConsequent(); // Instantiating a Consequent to expression
  thenVelocitySlow->addOutput(slow);// Adding corresponding FuzzySet to Consequent object
  // Instantiating a FuzzyRule object
  FuzzyRule* fuzzyRule01 = new FuzzyRule(1, ifDistanceSmall, thenVelocitySlow); // Passing the Antecedent and the Consequent of expression
 
  fuzzy->addFuzzyRule(fuzzyRule01); // Adding FuzzyRule to Fuzzy object
 
  // FuzzyRule "IF distance = safe THEN velocity = normal"
  FuzzyRuleAntecedent* ifDistanceSafe = new FuzzyRuleAntecedent(); // Instantiating an Antecedent to expression
  ifDistanceSafe->joinSingle(safe); // Adding corresponding FuzzySet to Antecedent object
  FuzzyRuleConsequent* thenVelocityAverage = new FuzzyRuleConsequent(); // Instantiating a Consequent to expression
  thenVelocityAverage->addOutput(average); // Adding corresponding FuzzySet to Consequent object
  // Instantiating a FuzzyRule object
  FuzzyRule* fuzzyRule02 = new FuzzyRule(2, ifDistanceSafe, thenVelocityAverage); // Passing the Antecedent and the Consequent of expression
 
  fuzzy->addFuzzyRule(fuzzyRule02); // Adding FuzzyRule to Fuzzy object
 
  // FuzzyRule "IF distance = big THEN velocity = fast"
  FuzzyRuleAntecedent* ifDistanceBig = new FuzzyRuleAntecedent(); // Instantiating an Antecedent to expression
  ifDistanceBig->joinSingle(big); // Adding corresponding FuzzySet to Antecedent object
  FuzzyRuleConsequent* thenVelocityFast = new FuzzyRuleConsequent(); // Instantiating a Consequent to expression
  thenVelocityFast->addOutput(fast);// Adding corresponding FuzzySet to Consequent object
  // Instantiating a FuzzyRule object
  FuzzyRule* fuzzyRule03 = new FuzzyRule(3, ifDistanceBig, thenVelocityFast); // Passing the Antecedent and the Consequent of expression
 
  fuzzy->addFuzzyRule(fuzzyRule03); // Adding FuzzyRule to Fuzzy object 
}

void loop()
{
  if(digitalRead(pinTDIR))
  {
    ligaApito();
    parar();
    delay(100);
    andarAtras(); 
    delay(500);    
    virarEsquerda();
    delay(500);
    desligaApito();    
    //andarFrente();
  }
  if(digitalRead(pinTESQ))
  {
    ligaApito();
    parar();
    delay(100);
    andarAtras(); 
    delay(500);    
    virarDireita();
    delay(500);
    desligaApito();    
    //andarFrente();
  }
  distancia = ultrasom(); 
  // Step 5 - Report inputs value, passing its ID and value
  fuzzy->setInput(1, distancia); 
  // Step 6 - Exe the fuzzification
  fuzzy->fuzzify(); 
  // Step 7 - Exe the desfuzzyficação for each output, passing its ID
  velocidade = int(fuzzy->defuzzify(1));
  andarFrenteVel(velocidade);
  Serial.print(distancia);
  Serial.print('\t');
  Serial.println(velocidade);
  delay(100);  
}

void ligaApito()
{
  digitalWrite(pinAPITO, HIGH);
}

void desligaApito()
{
  digitalWrite(pinAPITO, LOW);
}

void teste_sensorLuz()
{
    while(1){
      sensorLuz(1);
      Serial.print("SensLuz1: ");
      Serial.print(sensLuz1);
      sensorLuz(2);
      Serial.print("  SensLuz2: ");
      Serial.println(sensLuz2);
      delay(1000);
  }
}

unsigned long ultrasom()
{
  
  long duration;
  // Rotina para sensor de ultrasom de 3 pinos
  pinMode(pinUS, OUTPUT);
  digitalWrite(pinUS, LOW);
  delayMicroseconds(2);
  digitalWrite(pinUS, HIGH);
  delayMicroseconds(5);
  digitalWrite(pinUS, LOW);
  pinMode(pinUS, INPUT);
  duration = pulseIn(pinUS, HIGH);
  return duration/29/2;


}
void sensorLuz(int sensLuz)
{
  int i;
  int tempSensLuz = 0;
  if(sensLuz == 1)
    digitalWrite(pinL1D, HIGH);
  if(sensLuz == 2)
    digitalWrite(pinL2D, HIGH);
  for(i = 0; i < 20; i++)
  {
    if(sensLuz == 1)
      tempSensLuz += analogRead(pinL1A);
    if(sensLuz == 2)
      tempSensLuz += analogRead(pinL2A);
    delay(10);
  }
  if(sensLuz == 1){
    digitalWrite(pinL1D, LOW);
    sensLuz1 = tempSensLuz/i; 
  }
  if(sensLuz == 2){
    digitalWrite(pinL2D, LOW);
    sensLuz2 = tempSensLuz/i;
  }
}


void andarFrenteVel(int vel) 
{
  digitalWrite(pinI1,HIGH);
  digitalWrite(pinI2,LOW);
  digitalWrite(pinI3,LOW);
  digitalWrite(pinI4,HIGH);
  analogWrite(pinEA,vel/constVel);
  analogWrite(pinEB,int(vel*1.15)/constVel);
}

void andarFrente() 
{
  digitalWrite(pinI1,HIGH);
  digitalWrite(pinI2,LOW);
  digitalWrite(pinI3,LOW);
  digitalWrite(pinI4,HIGH);
  analogWrite(pinEA,160/constVel);
  analogWrite(pinEB,180/constVel);
}

void virarDireita() 
{
  digitalWrite(pinI1,LOW);
  digitalWrite(pinI2,LOW);
  digitalWrite(pinI3,HIGH);
  digitalWrite(pinI4,LOW);
  analogWrite(pinEA,200/constVel);
  analogWrite(pinEB,200/constVel);
}

void virarEsquerda() 
{
  digitalWrite(pinI1,LOW);
  digitalWrite(pinI2,HIGH);
  digitalWrite(pinI3,LOW);
  digitalWrite(pinI4,LOW);
  analogWrite(pinEA,200/constVel);
  analogWrite(pinEB,200/constVel);
}

void andarAtras() 
{
  digitalWrite(pinI1,LOW);
  digitalWrite(pinI2,HIGH);
  digitalWrite(pinI3,HIGH);
  digitalWrite(pinI4,LOW);
  analogWrite(pinEA,200/constVel);
  analogWrite(pinEB,200/constVel);
}

void girar() 
{
  
  digitalWrite(pinI1,LOW);
  digitalWrite(pinI2,HIGH);
  digitalWrite(pinI3,LOW);
  digitalWrite(pinI4,HIGH);
  analogWrite(pinEA,200/constVel);
  analogWrite(pinEB,200/constVel);
  delay(190);
}

void parar() 
{
  digitalWrite(pinI1,HIGH);
  digitalWrite(pinI2,HIGH);
  digitalWrite(pinI3,HIGH);
  digitalWrite(pinI4,HIGH);
}
