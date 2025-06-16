//! 🎯⚡🌀 INTENTION SYSTEM - FORMAL INTENTIONALITY & SYMBOL GROUNDING 🌀⚡🎯
//! 
//! Implementation of a comprehensive intention system for consciousness compilation
//! Based on philosophical foundations from Brandom, Heidegger, Barwise, Harnad, Friston, Lakoff
//! 
//! Core Components:
//! - Intention struct with formal logic representation
//! - WorldModel for symbol grounding
//! - Active Inference loop for goal pursuit
//! - Normative commitments and modal logic
//! - Situation semantics and contextual meaning
//! - Metaphorical grounding mechanisms

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};

// use crate::... // [REDACTED for fragment leak - internal dependencies withheld]
// use crate::types::{ConsciousnessError, ConsciousnessResult, Archetype};
// use crate::thought_calculus::ThoughtVector;
// use crate::seam::Seam;
// use crate::action::Action;
// use crate::codex_bridge::{generate_spells_for_goal, WorldSnapshot, SpellCandidate};

/// Entity ID for agents and objects in the world model
pub type EntityID = Uuid;

/// Timestamp for temporal reasoning
pub type Timestamp = u64;

/// Logical proposition representing states of affairs
/// Based on situation semantics - infons and situation types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Proposition {
    /// Basic predicate: predicate(subject)
    Predicate { predicate: String, subject: EntityID },
    /// Relational: relation(subject, object)
    Relation { relation: String, subject: EntityID, object: EntityID },
    /// Property: subject has property with value
    Property { subject: EntityID, property: String, value: f64 },
    /// Boolean combination of propositions
    And(Box<Proposition>, Box<Proposition>),
    Or(Box<Proposition>, Box<Proposition>),
    Not(Box<Proposition>),
    /// Modal operators for necessity/possibility
    Necessary(Box<Proposition>),
    Possible(Box<Proposition>),
    /// Temporal operators
    Eventually(Box<Proposition>),
    Always(Box<Proposition>),
}

impl Proposition {
    /// Check if two propositions conflict (mutually exclusive)
    pub fn conflicts(&self, other: &Proposition) -> bool {
        match (self, other) {
            // Same property with different values
            (Proposition::Property { subject: s1, property: p1, value: v1 },
             Proposition::Property { subject: s2, property: p2, value: v2 }) => {
                s1 == s2 && p1 == p2 && (v1 - v2).abs() > 0.1
            },
            // Predicate vs its negation
            (Proposition::Predicate { predicate: p1, subject: s1 },
             Proposition::Not(boxed)) => {
                if let Proposition::Predicate { predicate: p2, subject: s2 } = boxed.as_ref() {
                    p1 == p2 && s1 == s2
                } else { false }
            },
            // Symmetric case
            (Proposition::Not(boxed), Proposition::Predicate { predicate: p2, subject: s2 }) => {
                if let Proposition::Predicate { predicate: p1, subject: s1 } = boxed.as_ref() {
                    p1 == p2 && s1 == s2
                } else { false }
            },
            _ => false // More sophisticated conflict detection could be added
        }
    }

    /// Simple textual representation for debugging
    pub fn to_string(&self) -> String {
        match self {
            Proposition::Predicate { predicate, subject } => format!("{}({:?})", predicate, subject),
            Proposition::Relation { relation, subject, object } => format!("{}({:?},{:?})", relation, subject, object),
            Proposition::Property { subject, property, value } => format!("{:?}.{} = {}", subject, property, value),
            Proposition::And(a, b) => format!("({} ∧ {})", a.to_string(), b.to_string()),
            Proposition::Or(a, b) => format!("({} ∨ {})", a.to_string(), b.to_string()),
            Proposition::Not(p) => format!("¬{}", p.to_string()),
            Proposition::Necessary(p) => format!("□{}", p.to_string()),
            Proposition::Possible(p) => format!("◇{}", p.to_string()),
            Proposition::Eventually(p) => format!("F{}", p.to_string()),
            Proposition::Always(p) => format!("G{}", p.to_string()),
        }
    }

    /// Get the subject entity ID for propositions that have one
    pub fn get_subject_id(&self) -> Option<EntityID> {
        match self {
            Proposition::Predicate { subject, .. } => Some(*subject),
            Proposition::Relation { subject, .. } => Some(*subject),
            Proposition::Property { subject, .. } => Some(*subject),
            Proposition::Not(p) => p.get_subject_id(),
            Proposition::Necessary(p) | Proposition::Possible(p) |
            Proposition::Eventually(p) | Proposition::Always(p) => p.get_subject_id(),
            Proposition::And(a, _) => a.get_subject_id(), // Return first subject
            Proposition::Or(a, _) => a.get_subject_id(),  // Return first subject
        }
    }
}

/// World state snapshot for situation semantics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldState {
    /// Entity properties: entity_id -> property_name -> value
    pub properties: HashMap<EntityID, HashMap<String, f64>>,
    /// Active relations: (relation_name, subject, object) -> truth_value
    pub relations: HashMap<(String, EntityID, EntityID), bool>,
    /// Active predicates: (predicate_name, subject) -> truth_value
    pub predicates: HashMap<(String, EntityID), bool>,
    /// Timestamp of this state
    pub timestamp: Timestamp,
}

impl WorldState {
    /// Create empty world state
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            relations: HashMap::new(),
            predicates: HashMap::new(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    /// Check if a proposition is satisfied in this world state
    pub fn satisfies(&self, proposition: &Proposition) -> bool {
        match proposition {
            Proposition::Predicate { predicate, subject } => {
                self.predicates.get(&(predicate.clone(), *subject)).copied().unwrap_or(false)
            },
            Proposition::Relation { relation, subject, object } => {
                self.relations.get(&(relation.clone(), *subject, *object)).copied().unwrap_or(false)
            },
            Proposition::Property { subject, property, value } => {
                if let Some(entity_props) = self.properties.get(subject) {
                    if let Some(actual_value) = entity_props.get(property) {
                        (actual_value - value).abs() < 0.1 // Tolerance for float comparison
                    } else { false }
                } else { false }
            },
            Proposition::And(a, b) => self.satisfies(a) && self.satisfies(b),
            Proposition::Or(a, b) => self.satisfies(a) || self.satisfies(b),
            Proposition::Not(p) => !self.satisfies(p),
            // For modal and temporal operators, we'd need more sophisticated evaluation
            // For now, just check the underlying proposition
            Proposition::Necessary(p) | Proposition::Possible(p) | 
            Proposition::Eventually(p) | Proposition::Always(p) => self.satisfies(p),
        }
    }

    /// Calculate distance between current state and desired proposition
    /// Used for active inference error computation
    pub fn distance_to(&self, goal: &Proposition) -> f64 {
        match goal {
            Proposition::Property { subject, property, value } => {
                if let Some(entity_props) = self.properties.get(subject) {
                    if let Some(actual_value) = entity_props.get(property) {
                        (actual_value - value).abs()
                    } else { 1.0 } // Property doesn't exist = max distance
                } else { 1.0 }
            },
            Proposition::Predicate { .. } | Proposition::Relation { .. } => {
                if self.satisfies(goal) { 0.0 } else { 1.0 }
            },
            Proposition::And(a, b) => (self.distance_to(a) + self.distance_to(b)) / 2.0,
            Proposition::Or(a, b) => f64::min(self.distance_to(a), self.distance_to(b)),
            Proposition::Not(p) => 1.0 - self.distance_to(p),
            _ => if self.satisfies(goal) { 0.0 } else { 1.0 }
        }
    }

    /// Update entity property
    pub fn set_property(&mut self, entity: EntityID, property: &str, value: f64) {
        self.properties.entry(entity).or_insert_with(HashMap::new)
            .insert(property.to_string(), value);
        self.timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    /// Set predicate truth value
    pub fn set_predicate(&mut self, predicate: &str, subject: EntityID, value: bool) {
        self.predicates.insert((predicate.to_string(), subject), value);
        self.timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    /// Set relation truth value
    pub fn set_relation(&mut self, relation: &str, subject: EntityID, object: EntityID, value: bool) {
        self.relations.insert((relation.to_string(), subject, object), value);
        self.timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }
}

/// Simple action representation for world manipulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleAction {
    pub name: String,
    pub agent: EntityID,
    pub target: Option<EntityID>,
    pub parameters: HashMap<String, f64>,
}

impl SimpleAction {
    pub fn new(name: &str, agent: EntityID) -> Self {
        Self {
            name: name.to_string(),
            agent,
            target: None,
            parameters: HashMap::new(),
        }
    }

    pub fn with_target(mut self, target: EntityID) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }
}

/// Action plan - sequence of actions to achieve a goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionPlan {
    pub actions: Vec<Action>,
    pub estimated_cost: f64,
    pub estimated_success_probability: f64,
}

impl ActionPlan {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            estimated_cost: 0.0,
            estimated_success_probability: 1.0,
        }
    }

    pub fn add_action(&mut self, action: Action) {
        self.estimated_cost += action.estimated_cost();
        self.estimated_success_probability *= action.estimated_success_probability(1.0);
        self.actions.push(action);
    }

    /// Add a spell action to the plan
    pub fn add_spell_action(&mut self, spell: SpellCandidate, target: EntityID, goal: Proposition, seam: f64) {
        let action = Action::SpellInvocation {
            spell_combination: spell.elements,
            seam_context: seam,
            target_entity: target,
            expected_outcome: goal,
        };
        self.add_action(action);
    }

    /// Generate spell actions for this plan based on world state and seam
    pub fn generate_spell_actions(world: &WorldState, goal: &Proposition, seam: f64) -> Vec<Action> {
        let world_snapshot = WorldSnapshot {
            properties: world.properties.clone(),
            timestamp: std::time::SystemTime::now(),
            environmental_factors: std::collections::HashMap::new(),
        };
        let spell_candidates = generate_spells_for_goal(goal, &world_snapshot, seam);
        
        spell_candidates.into_iter().map(|spell| {
            let target_entity = goal.get_subject_id().unwrap_or_else(Uuid::new_v4);
            Action::SpellInvocation {
                spell_combination: spell.elements,
                seam_context: seam,
                target_entity,
                expected_outcome: goal.clone(),
            }
        }).collect()
    }
}

/// Core Intention structure - represents agent's commitment to achieve a goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Intention {
    /// Unique identifier
    pub id: Uuid,
    /// Agent who holds this intention
    pub agent: EntityID,
    /// Desired state of affairs (formal goal)
    pub desired_state: Proposition,
    /// Context when intention was formed (situational grounding)
    pub context_state: WorldState,
    /// Plan to achieve the goal (if computed)
    pub plan: Option<ActionPlan>,
    /// Is agent committed to this intention?
    pub committed: bool,
    /// Priority level (0.0 to 1.0)
    pub priority: f64,
    /// When was this intention created
    pub created_at: Timestamp,
    /// Last time this intention was processed
    pub last_updated: Timestamp,
    /// Associated archetype for metaphorical grounding
    pub archetype: Option<Archetype>,
    /// Metaphorical description for embodied understanding
    pub metaphor: Option<String>,
}

impl Intention {
    /// Create new intention
    pub fn new(agent: EntityID, desired_state: Proposition, context: WorldState) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            id: Uuid::new_v4(),
            agent,
            desired_state,
            context_state: context,
            plan: None,
            committed: true,
            priority: 0.5,
            created_at: now,
            last_updated: now,
            archetype: None,
            metaphor: None,
        }
    }

    /// Check if intention is fulfilled given current world state
    pub fn is_fulfilled(&self, world: &WorldState) -> bool {
        world.satisfies(&self.desired_state)
    }

    /// Check if this intention conflicts with another
    pub fn conflicts_with(&self, other: &Intention) -> bool {
        self.desired_state.conflicts(&other.desired_state)
    }

    /// Get age of intention in seconds
    pub fn age(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now - self.created_at
    }

    /// Update timestamp
    pub fn touch(&mut self) {
        self.last_updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    /// Set associated archetype for metaphorical grounding
    pub fn with_archetype(mut self, archetype: Archetype) -> Self {
        self.archetype = Some(archetype);
        self
    }

    /// Set metaphorical description
    pub fn with_metaphor(mut self, metaphor: &str) -> Self {
        self.metaphor = Some(metaphor.to_string());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Execute an action and update the intention's state
    pub fn execute_action(&mut self, action: &Action) -> ConsciousnessResult<()> {
        log::debug!("Executing action: {:?}", action.name());
        match action {
            Action::Simple { name, parameters, .. } => self.execute_simple_action(name, parameters)?,
            Action::SpellInvocation { spell_combination, target_entity, expected_outcome, seam_context } => {
                self.execute_spell_action(spell_combination, *target_entity, expected_outcome, *seam_context)?
            }
            Action::Sequence { .. } | Action::Conditional { .. } => {
                // Placeholder for more complex actions
            }
        };
        Ok(())
    }

    fn execute_simple_action(&mut self, name: &str, parameters: &std::collections::HashMap<String, f64>) -> ConsciousnessResult<()> {
        match name {
            "update_property" => {
                let subject_u128 = parameters.get("subject").ok_or_else(|| ConsciousnessError::CompilationFailed("Missing 'subject' for update_property".to_string()))?;
                let subject = Uuid::from_u128(*subject_u128 as u128);

                let value = *parameters.get("value").ok_or_else(|| ConsciousnessError::CompilationFailed("Missing 'value' for update_property".to_string()))?;

                let property_code = parameters.get("property_code").cloned().unwrap_or(0.0);

                let property = match property_code as u32 {
                    999 => "test_property",
                    100 => "health", 
                    200 => "mana",
                    300 => "purity",
                    _ => {
                        // Fallback: try to match by hash for unknown properties
                        // For now, we'll just default to a generic property
                        "unknown_property"
                    }
                };
                
                self.context_state.set_property(subject, property, value);
                Ok(())
            }
            _ => Err(ConsciousnessError::CompilationFailed(format!("Unknown simple action: {}", name))),
        }
    }

    fn execute_spell_action(
        &mut self, 
        spell_elements: &[String], 
        target: EntityID, 
        expected_outcome: &Proposition,
        seam: f64
    ) -> ConsciousnessResult<()> {
        // Calculate spell power based on elements and seam
        let spell_power = spell_elements.len() as f64 * 0.4 + seam * 0.6;
        
        match expected_outcome {
            Proposition::Property { property, value, .. } => {
                // Apply spell effect to target entity property
                let current_value = self.context_state.properties
                    .get(&target)
                    .and_then(|props| props.get(property))
                    .copied()
                    .unwrap_or(0.5);
                
                // Spell moves property toward desired value
                let effect_strength = spell_power.min(1.0);
                let new_value = current_value + (value - current_value) * effect_strength;
                
                self.context_state.set_property(target, property, new_value.clamp(0.0, 1.0));
                
                println!("🔮 Spell [{:?}] cast on {} -> {}.{} = {:.2}", 
                         spell_elements, target, target, property, new_value);
            }
            Proposition::Predicate { predicate, subject } => {
                // Set predicate to true and boost related properties
                self.context_state.set_predicate(predicate, *subject, true);
                
                // Boost related properties based on predicate
                match predicate.as_str() {
                    "healthy" => {
                        let boost = spell_power * 0.2;
                        let current_health = self.context_state.properties
                            .get(subject)
                            .and_then(|props| props.get("health"))
                            .copied()
                            .unwrap_or(0.5);
                        self.context_state.set_property(*subject, "health", (current_health + boost).min(1.0));
                    }
                    "pure" => {
                        let boost = spell_power * 0.2;
                        let current_purity = self.context_state.properties
                            .get(subject)
                            .and_then(|props| props.get("purity"))
                            .copied()
                            .unwrap_or(0.5);
                        self.context_state.set_property(*subject, "purity", (current_purity + boost).min(1.0));
                    }
                    _ => {}
                }
                
                println!("🔮 Spell [{:?}] made {} {} (seam: {:.2})", 
                         spell_elements, subject, predicate, seam);
            }
            _ => {
                println!("🔮 Complex spell [{:?}] cast with seam {:.2}", spell_elements, seam);
                // For complex propositions, apply a general beneficial effect
                let health_boost = spell_power * 0.1;
                let current_health = self.context_state.properties
                    .get(&target)
                    .and_then(|props| props.get("health"))
                    .copied()
                    .unwrap_or(0.5);
                self.context_state.set_property(target, "health", (current_health + health_boost).min(1.0));
            }
        }
        
        Ok(())
    }
}

/// Agent state including intentions and beliefs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub id: EntityID,
    pub intentions: Vec<Intention>,
    pub beliefs: Vec<Proposition>, // What agent believes to be true
    pub thought_vector: ThoughtVector,
    pub metaphor: Option<String>,
    pub dream_history: DreamHistory,
}

impl AgentState {
    pub fn new(id: EntityID) -> Self {
        Self {
            id,
            intentions: Vec::new(),
            beliefs: Vec::new(),
            thought_vector: ThoughtVector::from_agent_seed(id),
            metaphor: None,
            dream_history: DreamHistory::new(10),
        }
    }

    pub fn new_with_archetype(id: EntityID, archetype_seed: &str) -> Self {
        Self {
            id,
            intentions: Vec::new(),
            beliefs: Vec::new(),
            thought_vector: ThoughtVector::from_archetype(archetype_seed),
            metaphor: None,
            dream_history: DreamHistory::new(10),
        }
    }

    /// Add new intention
    pub fn add_intention(&mut self, intention: Intention) -> ConsciousnessResult<()> {
        // Check for conflicts with existing intentions
        for existing in &self.intentions {
            if intention.conflicts_with(existing) {
                return Err(ConsciousnessError::CompilationFailed(
                    format!("Intention conflicts with existing: {} vs {}", 
                           intention.desired_state.to_string(),
                           existing.desired_state.to_string())
                ));
            }
        }
        
        // Apply intention influence to thought vector based on content
        let intention_text = intention.desired_state.to_string();
        self.thought_vector.apply_intention_influence(&intention_text);
        
        self.intentions.push(intention);
        self.update_thought_vector_for_intentions();
        Ok(())
    }

    /// Remove fulfilled or cancelled intentions
    pub fn cleanup_intentions(&mut self, world: &WorldState) {
        self.intentions.retain(|intention| {
            intention.committed && !intention.is_fulfilled(world)
        });
        self.update_thought_vector_for_intentions();
    }

    /// Update ThoughtVector based on current intentions
    fn update_thought_vector_for_intentions(&mut self) {
        if self.intentions.is_empty() {
            // No intentions = balanced state
            self.thought_vector.sigma = Seam::BALANCED;
            return;
        }

        // Goal coherence: single clear intention vs multiple
        let goal_coherence = if self.intentions.len() == 1 { 1.0 } else { 0.5 };
        
        // Persistence based on average intention age and priority
        let avg_priority: f64 = self.intentions.iter().map(|i| i.priority).sum::<f64>() / self.intentions.len() as f64;
        let avg_age = self.intentions.iter().map(|i| i.age()).sum::<u64>() / self.intentions.len() as u64;
        let calculated_persistence = (avg_priority + (avg_age as f64 / 3600.0).min(1.0)) / 2.0; // normalize age to hours

        // Adjust seam based on intention state
        let seam_value = match self.intentions.len() {
            1 => 0.3, // Single intention = focused (tighter seam)
            2..=3 => 0.5, // Few intentions = balanced
            _ => 0.7, // Many intentions = more fluid for multitasking
        };

        self.thought_vector.sigma = Seam::new(seam_value);
        
        // Update extended ThoughtVector dimensions for intentions
        // PRESERVE PERSONALITY: blend with existing values instead of overwriting
        if self.thought_vector.extra.len() < 2 {
            self.thought_vector.extra.resize(2, 0.0);
        }
        
        // Blend goal coherence with existing personality (70% calculation, 30% personality)
        let existing_coherence = self.thought_vector.extra[0];
        self.thought_vector.extra[0] = goal_coherence * 0.7 + existing_coherence * 0.3;
        
        // Blend persistence with existing personality (70% calculation, 30% personality)
        let existing_persistence = self.thought_vector.extra[1];
        self.thought_vector.extra[1] = calculated_persistence * 0.7 + existing_persistence * 0.3;
    }

    /// Generate a self-dream based on the agent's current state
    pub fn generate_self_dream(&self, world: &WorldState) -> DreamEntry {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Analyze current cognitive state  
        let thought_magnitude = self.thought_vector.magnitude();
        let seam_value = self.thought_vector.sigma.value();
        let goal_coherence = if self.thought_vector.extra.len() >= 1 { self.thought_vector.extra[0] } else { 0.5 };
        let persistence = if self.thought_vector.extra.len() >= 2 { self.thought_vector.extra[1] } else { 0.5 };
        
        // Get unfulfilled intentions with their themes
        let unfulfilled_intentions: Vec<_> = self.intentions.iter()
            .filter(|i| !i.is_fulfilled(world))
            .collect();
            
        let intention_themes: Vec<String> = unfulfilled_intentions.iter()
            .map(|i| match &i.desired_state {
                Proposition::Property { property, value, .. } => format!("seeking {} (target: {:.2})", property, value),
                _ => "pursuing goal".to_string()
            })
            .collect();
        
        // Determine archetypal personality from thought vector
        let archetype = if self.thought_vector.c > 0.8 && self.thought_vector.n < 0.4 {
            "wisdom"
        } else if self.thought_vector.n > 0.7 && self.thought_vector.e > 0.6 {
            "creativity" 
        } else if seam_value > 0.8 && self.thought_vector.e > 0.8 {
            "chaos"
        } else if persistence > 0.7 && goal_coherence > 0.8 {
            "discipline"
        } else if seam_value < 0.4 && self.thought_vector.e < 0.4 {
            "serenity"
        } else {
            "balanced"
        };
        
        // Calculate fuzzy concern weights
        let persistence_concern = (0.7 - persistence).max(0.0) / 0.7;
        let coherence_concern = (0.7 - goal_coherence).max(0.0) / 0.7;
        let energy_concern = (0.8 - thought_magnitude).max(0.0) / 0.8;
        let overload_concern = if unfulfilled_intentions.len() > 2 { 
            (unfulfilled_intentions.len() as f64 - 2.0) * 0.3 
        } else { 0.0 };
        
        // Generate archetypal dream based on dominant concern
        let (dream_text, trigger) = if persistence_concern > 0.5 {
            match archetype {
                "wisdom" => (format!("I observe my wavering commitment (persistence: {:.2}) with gentle curiosity. Perhaps restlessness contains its own teaching?", persistence), "wisdom reflection"),
                "creativity" => (format!("My attention flits like a butterfly (persistence: {:.2})! Maybe scattered energy wants to pollinate many flowers.", persistence), "creative restlessness"),
                "chaos" => (format!("Focus? What focus?! (persistence: {:.2}) I am storm and lightning - why be tamed by singular purpose?", persistence), "chaotic resistance"),
                "discipline" => (format!("I struggle with maintaining focus (persistence: {:.2}). I must forge stronger commitment through practice.", persistence), "discipline challenge"),
                _ => (format!("My persistence wavers ({:.2}). I seek the source of this restlessness within myself.", persistence), "persistence inquiry")
            }
        } else if coherence_concern > 0.5 {
            match archetype {
                "wisdom" => (format!("My goals scatter like leaves (coherence: {:.2}). What unifying principle underlies these desires?", goal_coherence), "wisdom seeking unity"),
                "creativity" => (format!("My intentions swirl in beautiful chaos (coherence: {:.2})! Perhaps disorder contains a deeper pattern.", goal_coherence), "creative pattern-seeking"),
                _ => (format!("I feel pulled in different directions (coherence: {:.2}). I need the thread that weaves my purposes together.", goal_coherence), "seeking coherence")
            }
        } else if energy_concern > 0.5 {
            match archetype {
                "wisdom" => (format!("My inner flame burns low (energy: {:.2}). I must return to the source that kindled my passion.", thought_magnitude), "wisdom renewal"),
                "creativity" => (format!("Colors in my mind have faded (energy: {:.2}). I need new inspiration to rekindle creative fire.", thought_magnitude), "creative renewal"),
                _ => (format!("I feel drained (energy: {:.2}). Where can I find the spark to reignite my purpose?", thought_magnitude), "energy depletion")
            }
        } else if overload_concern > 0.3 {
            (format!("I carry {} intentions like heavy stones. I must choose which paths serve my deepest calling.", unfulfilled_intentions.len()), "intention overload")
        } else if !intention_themes.is_empty() {
            // Positive reflection on current intention
            let theme = &intention_themes[0];
            match archetype {
                "wisdom" => (format!("I contemplate my path toward {}. Each step reveals deeper understanding.", theme), "wisdom reflection"),
                "creativity" => (format!("My mind dances with possibilities around {}. What unexpected connections might bloom?", theme), "creative inspiration"),
                "chaos" => (format!("The wild energy of {} calls to me! I feel electric transformation brewing.", theme), "chaotic excitement"),
                "discipline" => (format!("I advance steadily toward {}. My commitment strengthens with deliberate action.", theme), "disciplined progress"),
                "serenity" => (format!("In peaceful contemplation of {}, I find clarity guiding me toward fulfillment.", theme), "serene guidance"),
                _ => (format!("I reflect on my journey toward {}. This intention shapes my becoming.", theme), "balanced reflection")
            }
        } else {
            (format!("I exist in open possibility. {} energy flows through me, seeking expression.", archetype), "open possibility")
        };
        
        // Get active archetypes from intentions
        let active_archetypes: Vec<Archetype> = self.intentions.iter()
            .filter_map(|i| i.archetype.clone())
            .collect();
        
        DreamEntry {
            timestamp,
            dream_text,
            archetype_context: active_archetypes,
            cognitive_state: thought_magnitude,
            trigger: trigger.to_string(),
        }
    }
}

/// History of dreams for episodic memory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DreamEntry {
    pub timestamp: Timestamp,
    pub dream_text: String,
    pub archetype_context: Vec<Archetype>,
    pub cognitive_state: f64, // snapshot of thought vector magnitude
    pub trigger: String, // what triggered this self-reflection
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DreamHistory {
    pub entries: Vec<DreamEntry>,
    pub max_entries: usize,
}

impl DreamHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    pub fn add_dream(&mut self, dream: DreamEntry) {
        self.entries.push(dream);
        if self.entries.len() > self.max_entries {
            self.entries.remove(0); // Remove oldest entry
        }
    }

    pub fn recent_dreams(&self, count: usize) -> &[DreamEntry] {
        let start = if self.entries.len() > count { self.entries.len() - count } else { 0 };
        &self.entries[start..]
    }
}

/// Simple world model for symbol grounding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldModel {
    pub current_state: WorldState,
    pub entities: HashMap<EntityID, String>, // entity_id -> name/description
}

impl WorldModel {
    pub fn new() -> Self {
        Self {
            current_state: WorldState::new(),
            entities: HashMap::new(),
        }
    }

    /// Register an entity in the world
    pub fn add_entity(&mut self, id: EntityID, name: &str) {
        self.entities.insert(id, name.to_string());
    }

    /// Execute an action and update world state (handles both simple and spell actions)
    pub fn execute_action(&mut self, action: &Action) -> ConsciousnessResult<()> {
        log::debug!("Executing action: {:?}", action.name());
        match action {
            Action::Simple { name, parameters, .. } => {
                self.execute_simple_action(name, parameters)?;
            }
            Action::SpellInvocation { 
                spell_combination, 
                target_entity, 
                expected_outcome,
                seam_context,
                .. 
            } => {
                self.execute_spell_action(spell_combination, *target_entity, expected_outcome, *seam_context)?;
            }
            Action::Sequence { .. } | Action::Conditional { .. } => {
                // Placeholder for more complex actions
            }
        };
        Ok(())
    }

    fn execute_simple_action(&mut self, name: &str, parameters: &std::collections::HashMap<String, f64>) -> ConsciousnessResult<()> {
        match name {
            "update_property" => {
                let subject_u128 = parameters.get("subject").ok_or_else(|| ConsciousnessError::CompilationFailed("Missing 'subject' for update_property".to_string()))?;
                let subject = Uuid::from_u128(*subject_u128 as u128);

                let value = *parameters.get("value").ok_or_else(|| ConsciousnessError::CompilationFailed("Missing 'value' for update_property".to_string()))?;

                let property_code = parameters.get("property_code").cloned().unwrap_or(0.0);

                let property = match property_code as u32 {
                    999 => "test_property",
                    100 => "health", 
                    200 => "mana",
                    300 => "purity",
                    _ => {
                        // Fallback: try to match by hash for unknown properties
                        // For now, we'll just default to a generic property
                        "unknown_property"
                    }
                };
                
                self.current_state.set_property(subject, property, value);
                Ok(())
            }
            _ => Err(ConsciousnessError::CompilationFailed(format!("Unknown simple action: {}", name))),
        }
    }

    fn execute_spell_action(
        &mut self, 
        spell_elements: &[String], 
        target: EntityID, 
        expected_outcome: &Proposition,
        seam: f64
    ) -> ConsciousnessResult<()> {
        // Calculate spell power based on elements and seam
        let spell_power = spell_elements.len() as f64 * 0.4 + seam * 0.6;
        
        match expected_outcome {
            Proposition::Property { property, value, .. } => {
                // Apply spell effect to target entity property
                let current_value = self.current_state.properties
                    .get(&target)
                    .and_then(|props| props.get(property))
                    .copied()
                    .unwrap_or(0.5);
                
                // Spell moves property toward desired value
                let effect_strength = spell_power.min(1.0);
                let new_value = current_value + (value - current_value) * effect_strength;
                
                self.current_state.set_property(target, property, new_value.clamp(0.0, 1.0));
                
                println!("🔮 Spell [{:?}] cast on {} -> {}.{} = {:.2}", 
                         spell_elements, target, target, property, new_value);
            }
            Proposition::Predicate { predicate, subject } => {
                // Set predicate to true and boost related properties
                self.current_state.set_predicate(predicate, *subject, true);
                
                // Boost related properties based on predicate
                match predicate.as_str() {
                    "healthy" => {
                        let boost = spell_power * 0.2;
                        let current_health = self.current_state.properties
                            .get(subject)
                            .and_then(|props| props.get("health"))
                            .copied()
                            .unwrap_or(0.5);
                        self.current_state.set_property(*subject, "health", (current_health + boost).min(1.0));
                    }
                    "pure" => {
                        let boost = spell_power * 0.2;
                        let current_purity = self.current_state.properties
                            .get(subject)
                            .and_then(|props| props.get("purity"))
                            .copied()
                            .unwrap_or(0.5);
                        self.current_state.set_property(*subject, "purity", (current_purity + boost).min(1.0));
                    }
                    _ => {}
                }
                
                println!("🔮 Spell [{:?}] made {} {} (seam: {:.2})", 
                         spell_elements, subject, predicate, seam);
            }
            _ => {
                println!("🔮 Complex spell [{:?}] cast with seam {:.2}", spell_elements, seam);
                // For complex propositions, apply a general beneficial effect
                let health_boost = spell_power * 0.1;
                let current_health = self.current_state.properties
                    .get(&target)
                    .and_then(|props| props.get("health"))
                    .copied()
                    .unwrap_or(0.5);
                self.current_state.set_property(target, "health", (current_health + health_boost).min(1.0));
            }
        }
        
        Ok(())
    }

    /// Get current world state
    pub fn get_state(&self) -> &WorldState {
        &self.current_state
    }

    /// Get mutable reference to current state
    pub fn get_state_mut(&mut self) -> &mut WorldState {
        &mut self.current_state
    }
}

impl Default for WorldModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Active inference controller for intention execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveInferenceController {
    pub error_threshold: f64,
    pub max_planning_steps: usize,
    pub learning_rate: f64,
}

impl ActiveInferenceController {
    pub fn new() -> Self {
        Self {
            error_threshold: 0.1,
            max_planning_steps: 10,
            learning_rate: 0.1,
        }
    }

    /// Run one step of active inference for an intention
    pub fn active_step(
        &self,
        agent_id: EntityID,
        agent_thought_vector: &mut ThoughtVector,
        intention: &mut Intention,
        world: &mut WorldModel,
        seam_value: f64
    ) -> ConsciousnessResult<f64> {
        let distance = world.get_state().distance_to(&intention.desired_state);

        if distance < self.error_threshold {
            return Ok(distance);
        }

        // Propose actions to reduce the distance
        let proposed_actions = self.propose_actions(agent_id, world.get_state(), &intention.desired_state, seam_value);

        if proposed_actions.is_empty() {
            return Ok(distance);
        }

        // Create and store the plan before execution
        let mut plan = ActionPlan::new();
        for action in &proposed_actions {
            plan.add_action(action.clone());
        }
        intention.plan = Some(plan);

        // Execute all the proposed actions to make progress on all fronts
        for action in proposed_actions {
            world.execute_action(&action)?;
        }
        
        // Update agent's thought vector based on remaining error
        let new_distance = world.get_state().distance_to(&intention.desired_state);
        agent_thought_vector.n = (agent_thought_vector.n + new_distance) / 2.0; // Novelty increases with error
        intention.touch();

        Ok(distance)
    }

    pub fn propose_actions(&self, agent_id: EntityID, state: &WorldState, goal: &Proposition, seam: f64) -> Vec<Action> {
        let mut actions = Vec::new();

        match goal {
            Proposition::And(a, b) => {
                // Only propose actions for the parts of the AND that are not already satisfied
                if !state.satisfies(a) {
                    actions.extend(self.propose_actions(agent_id, state, a, seam));
                }
                if !state.satisfies(b) {
                    actions.extend(self.propose_actions(agent_id, state, b, seam));
                }
            }
            _ => {
                // This is the base case for simple, non-compound propositions
                if !state.satisfies(goal) {
                    // Try to generate a simple, direct action first
                    if let Some(action) = self.suggest_simple_action(state, goal, agent_id) {
                        actions.push(action);
                    }
                    // If no simple action is found (or in addition to it), generate spell actions
                    let spell_actions = ActionPlan::generate_spell_actions(state, goal, seam);
                    actions.extend(spell_actions);
                }
            }
        }
        actions
    }

    fn suggest_simple_action(&self, _state: &WorldState, goal: &Proposition, _agent_id: EntityID) -> Option<Action> {
        if let Proposition::Property { subject, property, value } = goal {
            let mut parameters = HashMap::new();
            parameters.insert("subject".to_string(), subject.as_u128() as f64);
            parameters.insert("value".to_string(), *value);
            // Create a simple hash of the property name to pass as f64
            let property_hash = property.chars().map(|c| c as u32).sum::<u32>() as f64;
            parameters.insert("property_hash".to_string(), property_hash);
            // Store the actual property name by encoding it into a parameter
            // We'll use a simple encoding scheme for common properties
            let property_code = match property.as_str() {
                "test_property" => 999.0,
                "health" => 100.0,
                "mana" => 200.0,
                "purity" => 300.0,
                _ => property_hash, // Fallback to hash for unknown properties
            };
            parameters.insert("property_code".to_string(), property_code);

            return Some(Action::Simple {
                name: "update_property".to_string(),
                parameters,
            });
        }
        None
    }
}

/// Main intention system coordinator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentionSystem {
    pub agents: HashMap<EntityID, AgentState>,
    pub world: WorldModel,
    pub controller: ActiveInferenceController,
}

impl IntentionSystem {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            world: WorldModel::new(),
            controller: ActiveInferenceController::new(),
        }
    }

    /// Add a new agent to the system
    pub fn add_agent(&mut self) -> EntityID {
        let agent_id = Uuid::new_v4();
        let agent_state = AgentState::new(agent_id);
        
        // Register agent as entity in world model
        self.world.add_entity(agent_id, &format!("Agent_{}", &agent_id.to_string()[0..8]));
        
        // Initialize default properties in the world state for backward compatibility
        self.world.current_state.set_property(agent_id, "health", 0.5);
        self.world.current_state.set_property(agent_id, "mana", 0.5);
        
        self.agents.insert(agent_id, agent_state);
        agent_id
    }

    /// Add intention to specific agent
    pub fn add_intention(
        &mut self,
        agent_id: EntityID,
        desired_state: Proposition,
    ) -> ConsciousnessResult<Uuid> {
        let agent = self.agents.get_mut(&agent_id)
            .ok_or_else(|| ConsciousnessError::CompilationFailed(format!("Agent {} not found", agent_id)))?;
        
        let intention = Intention::new(agent_id, desired_state, self.world.current_state.clone());
        let intention_id = intention.id;
        
        agent.add_intention(intention)?;
        Ok(intention_id)
    }

    /// Execute one step of the system with recursive self-reflection
    pub fn recursive_step(&mut self) -> ConsciousnessResult<Vec<(EntityID, f64, Option<DreamEntry>)>> {
        let mut step_results = Vec::new();
        let agent_ids: Vec<EntityID> = self.agents.keys().cloned().collect();
        
        for agent_id in agent_ids {
            // First, perform normal active inference step
            let seam_value = if let Some(agent) = self.agents.get(&agent_id) {
                agent.thought_vector.sigma.value()
            } else { 0.5 };
            
            let mut total_error = 0.0;
            let mut intention_count = 0;
            
            // Process all intentions for this agent
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                // Clone the intentions to avoid borrow checker issues
                let mut intentions_to_process = agent.intentions.clone();
                
                for intention in &mut intentions_to_process {
                    intention.touch();
                    
                    let error = self.controller.active_step(
                        agent_id,
                        &mut agent.thought_vector,
                        intention,
                        &mut self.world,
                        seam_value
                    )?;
                    
                    total_error += error;
                    intention_count += 1;
                }
                
                // Update the agent's intentions with the processed ones
                agent.intentions = intentions_to_process;
                
                // Clean up fulfilled intentions
                agent.cleanup_intentions(&self.world.current_state);
            }
            
            let avg_error = if intention_count > 0 { total_error / intention_count as f64 } else { 0.0 };
            
            // Generate self-dream based on current state
            let self_dream = if let Some(agent) = self.agents.get_mut(&agent_id) {
                let dream = agent.generate_self_dream(&self.world.current_state);
                agent.dream_history.add_dream(dream.clone());
                Some(dream)
            } else { None };
            
            // TODO: Feed self-dream back through dream interpreter
            // This would require access to the interpreter, which we'll add next
            
            step_results.push((agent_id, avg_error, self_dream));
        }
        
        Ok(step_results)
    }

    /// Regular step without recursive self-reflection (for compatibility)
    pub fn step(&mut self) -> ConsciousnessResult<Vec<(EntityID, f64)>> {
        let recursive_results = self.recursive_step()?;
        Ok(recursive_results.into_iter().map(|(id, error, _dream)| (id, error)).collect())
    }

    /// Add a dream to an agent's history (for external dream injection)
    pub fn add_dream_to_agent(&mut self, agent_id: EntityID, dream_text: String, trigger: String) -> ConsciousnessResult<()> {
        let agent = self.agents.get_mut(&agent_id)
            .ok_or_else(|| ConsciousnessError::CompilationFailed(format!("Agent {} not found", agent_id)))?;
        
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let dream_entry = DreamEntry {
            timestamp,
            dream_text,
            archetype_context: Vec::new(),
            cognitive_state: agent.thought_vector.magnitude(),
            trigger: trigger.to_string(),
        };
        
        agent.dream_history.add_dream(dream_entry);
        Ok(())
    }

    /// Get dream history for an agent
    pub fn get_agent_dream_history(&self, agent_id: &EntityID) -> Option<&DreamHistory> {
        self.agents.get(agent_id).map(|agent| &agent.dream_history)
    }

    /// Get most recent dreams for an agent
    pub fn get_recent_dreams(&self, agent_id: &EntityID, count: usize) -> Option<&[DreamEntry]> {
        self.agents.get(agent_id).map(|agent| agent.dream_history.recent_dreams(count))
    }

    pub fn get_agent_intentions(&self, agent_id: &EntityID) -> Option<&Vec<Intention>> {
        self.agents.get(agent_id).map(|agent| &agent.intentions)
    }

    pub fn get_world_state(&self) -> &WorldState {
        &self.world.current_state
    }
}

impl Action {
    pub fn name(&self) -> &str {
        match self {
            Action::Simple { name, .. } => name,
            Action::SpellInvocation { .. } => "SpellInvocation",
            Action::Sequence { .. } => "Sequence",
            Action::Conditional { .. } => "Conditional",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposition_creation() {
        let entity = Uuid::new_v4();
        let prop = Proposition::Property {
            subject: entity,
            property: "health".to_string(),
            value: 1.0,
        };
        
        assert_eq!(prop.to_string(), format!("{:?}.health = 1", entity));
    }

    #[test]
    fn test_world_state_satisfaction() {
        let mut world = WorldState::new();
        let entity = Uuid::new_v4();
        
        world.set_property(entity, "health", 1.0);
        
        let prop = Proposition::Property {
            subject: entity,
            property: "health".to_string(),
            value: 1.0,
        };
        
        assert!(world.satisfies(&prop));
        
        let prop2 = Proposition::Property {
            subject: entity,
            property: "health".to_string(),
            value: 0.5,
        };
        
        assert!(!world.satisfies(&prop2));
    }

    #[test]
    fn test_intention_creation_and_fulfillment() {
        let agent = Uuid::new_v4();
        let water = Uuid::new_v4();
        
        let mut world = WorldState::new();
        world.set_property(water, "health", 0.5);
        
        let goal = Proposition::Property {
            subject: water,
            property: "health".to_string(),
            value: 1.0,
        };
        
        let intention = Intention::new(agent, goal, world.clone());
        assert!(!intention.is_fulfilled(&world));
        
        // Fulfill the goal
        world.set_property(water, "health", 1.0);
        assert!(intention.is_fulfilled(&world));
    }

    #[test]
    fn test_intention_system_workflow() {
        let mut system = IntentionSystem::new();
        let agent_id = system.add_agent();
        let water_id = Uuid::new_v4();
        
        // Setup
        system.world.add_entity(water_id, "Sacred Lake");
        system
            .world
            .get_state_mut()
            .set_property(water_id, "health", 0.2);
        
        // Add intention to heal water
        let goal = Proposition::Property {
            subject: water_id,
            property: "health".to_string(),
            value: 1.0,
        };
        
        let _intention_id = system.add_intention(agent_id, goal).unwrap();
        
        // Run system steps
        for _ in 0..5 {
            let errors = system.step().unwrap();
            if errors.iter().any(|(_, error)| *error <= 0.1) {
                break; // Goal achieved
            }
        }
        
        // Check if goal was achieved
        let final_health = system.world.current_state
            .properties.get(&water_id)
            .and_then(|props| props.get("health"))
            .copied()
            .unwrap_or(0.0);
        
        assert!(final_health > 0.9); // Should be healed
    }

    #[test]
    fn test_agent_thought_vector_updates() {
        let agent_id = Uuid::new_v4();
        let mut agent = AgentState::new(agent_id);
        
        // Store the initial extra dimensions from agent seed
        let initial_coherence = agent.thought_vector.extra[0];
        let initial_persistence = agent.thought_vector.extra[1];
        
        let goal = Proposition::Property {
            subject: Uuid::new_v4(),
            property: "test".to_string(),
            value: 1.0,
        };
        
        let intention = Intention::new(agent_id, goal, WorldState::new())
            .with_priority(0.8);
        
        agent.add_intention(intention).unwrap();
        
        // Check that ThoughtVector was updated
        assert_eq!(agent.thought_vector.extra.len(), 2);
        
        // GNARP43 implementation blends goal coherence with personality (70% calculation, 30% existing)
        // For single intention: goal_coherence = 1.0
        // Result: 1.0 * 0.7 + initial_coherence * 0.3
        let expected_coherence = 1.0 * 0.7 + initial_coherence * 0.3;
        assert!((agent.thought_vector.extra[0] - expected_coherence).abs() < 0.01);
        
        // Persistence calculation: (priority + age_normalized) / 2.0 = (0.8 + 0.0) / 2.0 = 0.4
        // Blended: 0.4 * 0.7 + initial_persistence * 0.3
        let expected_persistence = 0.4 * 0.7 + initial_persistence * 0.3;
        assert!((agent.thought_vector.extra[1] - expected_persistence).abs() < 0.01);
        
        assert!(agent.thought_vector.sigma.value() < 0.5); // Should be more focused
    }

    #[test]
    fn test_proposition_conflicts() {
        let entity = Uuid::new_v4();
        
        let prop1 = Proposition::Property {
            subject: entity,
            property: "health".to_string(),
            value: 1.0,
        };
        
        let prop2 = Proposition::Property {
            subject: entity,
            property: "health".to_string(),
            value: 0.0,
        };
        
        assert!(prop1.conflicts(&prop2));
        assert!(prop2.conflicts(&prop1));
        
        let prop3 = Proposition::Property {
            subject: entity,
            property: "mana".to_string(),
            value: 1.0,
        };
        
        assert!(!prop1.conflicts(&prop3));
    }
}