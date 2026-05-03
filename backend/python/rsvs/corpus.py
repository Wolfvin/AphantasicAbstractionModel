"""
RSVS v0.9 — Embedded corpus for demonstration and benchmarking.

Wikipedia-style texts covering multiple domains.
Used when live Wikipedia access is unavailable.
Each domain is a list of sentences extracted/paraphrased from Wikipedia articles.
"""

DOMAINS = {

    "geology": [
        "Rock is a naturally occurring solid aggregate of one or more minerals.",
        "Stone is a hard solid mineral material found in the earth.",
        "Granite is a coarse-grained igneous rock composed mainly of quartz and feldspar.",
        "Basalt is a dark fine-grained volcanic rock that is the most common type of solid rock.",
        "Sedimentary rock is formed by the deposition and cementation of mineral particles.",
        "Limestone is a sedimentary rock composed mainly of calcium carbonate.",
        "Marble is a metamorphic rock composed of recrystallized carbonate minerals.",
        "Quartz is the second most abundant mineral in the continental crust.",
        "Feldspar is the most abundant mineral group in the earth's crust.",
        "Erosion is the process by which rock and soil are worn away by wind and water.",
        "Weathering breaks down rock through physical and chemical processes over time.",
        "A mineral is a naturally occurring inorganic solid with a crystalline structure.",
        "Crystals form when a mineral solidifies slowly from a liquid state.",
        "Volcanic rock is formed when magma erupts from a volcano and cools rapidly.",
        "Igneous rock forms through the cooling and solidification of magma or lava.",
        "Metamorphic rock forms when existing rock is changed by heat and pressure.",
        "The rock cycle describes how rocks change from one type to another over time.",
        "Sandstone is a sedimentary rock composed of sand-sized mineral particles.",
        "Shale is a fine-grained sedimentary rock formed from clay and silt particles.",
        "Obsidian is a naturally occurring volcanic glass formed from rapidly cooling lava.",
        "Coal is a combustible black sedimentary rock formed from plant material.",
        "Diamond is the hardest natural material and a form of carbon crystal.",
        "Gold is a dense soft shiny metallic element that is highly resistant to corrosion.",
        "Iron is the most common element on Earth by mass and a hard metallic substance.",
        "Copper is a soft malleable ductile metal with very high thermal and electrical conductivity.",
        "A fossil is the preserved remains of a once-living organism found in rock.",
        "Tectonic plates are large segments of the earth's crust that move slowly over time.",
        "An earthquake occurs when tectonic plates suddenly shift releasing stored energy.",
        "A volcano is an opening in the earth's crust through which lava and gases escape.",
        "Soil is a mixture of organic matter minerals gases liquids and organisms.",
    ],

    "water": [
        "Water is a transparent tasteless odorless nearly colorless chemical substance.",
        "Water is composed of hydrogen and oxygen atoms bonded together.",
        "Ice is the solid form of water and forms when water freezes below zero degrees.",
        "Steam is the gaseous form of water produced when water is heated to boiling.",
        "A river is a large natural flowing watercourse moving toward an ocean or lake.",
        "An ocean is a vast body of salt water covering most of the earth's surface.",
        "Rain is liquid water falling from clouds in the atmosphere.",
        "Snow is precipitation in the form of ice crystals falling from clouds.",
        "A lake is a large body of water surrounded by land.",
        "A waterfall is a point where water flows over a vertical drop in a river.",
        "Evaporation is the process by which liquid water transforms into water vapor.",
        "Condensation is the process by which water vapor transforms into liquid water.",
        "The water cycle describes the continuous movement of water on Earth.",
        "Groundwater is water found underground in the cracks and spaces in soil and rock.",
        "A glacier is a large persistent body of dense ice that moves slowly over land.",
        "Water is essential for all known forms of life to survive and function.",
        "Freshwater is water with low concentrations of dissolved salts.",
        "Saltwater contains dissolved salts and minerals making it unfit for drinking.",
        "Waves are the movement of energy through water caused by wind or geological activity.",
        "Tides are the regular rise and fall of sea levels caused by the moon's gravity.",
        "A flood occurs when water overflows its normal boundaries and inundates land.",
        "Drought is a prolonged period of abnormally low rainfall.",
        "Humidity is the amount of water vapor present in the air.",
        "Aquatic means living or growing in or near water.",
        "A wetland is a land area that is saturated with water either permanently or seasonally.",
        "Coral reefs are underwater structures made of calcium carbonate secreted by corals.",
        "Pressure increases with depth in water due to the weight of the water above.",
        "Surface tension is the tendency of liquid surfaces to shrink to the minimum area.",
        "Water is an excellent solvent and dissolves many substances readily.",
        "The density of water is approximately one gram per cubic centimeter.",
    ],

    "biology": [
        "A cell is the basic structural and functional unit of all known living organisms.",
        "DNA is a molecule that carries the genetic instructions for living organisms.",
        "Proteins are large complex molecules that carry out most of the work in cells.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "Respiration is the process by which organisms convert oxygen and glucose into energy.",
        "Evolution is the change in heritable traits of populations over successive generations.",
        "Natural selection is the mechanism by which organisms with favorable traits reproduce more.",
        "A species is a group of organisms capable of interbreeding and producing fertile offspring.",
        "An ecosystem is a community of living organisms interacting with their environment.",
        "A food chain shows how energy flows from one organism to another through feeding.",
        "A predator is an organism that hunts and feeds on other organisms called prey.",
        "Bacteria are single-celled microorganisms that are among the earliest life forms on Earth.",
        "A virus is an infectious agent that replicates inside living cells of organisms.",
        "Fungi are a kingdom of organisms including molds yeasts and mushrooms.",
        "Plants are multicellular organisms that produce energy through photosynthesis.",
        "Animals are multicellular organisms that consume food for energy and movement.",
        "Mammals are warm-blooded vertebrates that nurse their young with milk.",
        "Birds are warm-blooded egg-laying vertebrates with feathers and beaks.",
        "Fish are cold-blooded aquatic vertebrates with gills and fins.",
        "Insects are the largest class of animals with three body segments and six legs.",
        "A habitat is the natural environment in which an organism lives.",
        "Biodiversity refers to the variety of life found on Earth.",
        "Extinction is the termination of a species when the last individual dies.",
        "A gene is a sequence of DNA that contains instructions for making a protein.",
        "Heredity is the passing of traits from parents to offspring through genes.",
        "A hormone is a chemical messenger produced by glands that regulates body functions.",
        "The nervous system controls and coordinates body functions through electrical signals.",
        "The immune system protects the body against pathogens and disease.",
        "Metabolism refers to all chemical reactions that occur in a living organism.",
        "A symbiosis is a close long-term interaction between two different biological species.",
    ],

    "physics": [
        "Force is an interaction that changes the motion of an object.",
        "Mass is the amount of matter contained in an object.",
        "Energy is the capacity to do work and exists in many forms.",
        "Heat is a form of energy that flows from a hot object to a cold one.",
        "Light is electromagnetic radiation that is visible to the human eye.",
        "Sound is a mechanical wave that propagates through a medium such as air.",
        "Gravity is the force that attracts objects with mass toward one another.",
        "Electricity is the flow of electric charge through a conductor.",
        "Magnetism is a force exerted by magnets and electric currents.",
        "Temperature is a measure of the average kinetic energy of particles.",
        "Pressure is force applied per unit area on a surface.",
        "Velocity is the rate of change of position with respect to time.",
        "Acceleration is the rate of change of velocity with respect to time.",
        "A wave is a disturbance that transfers energy through matter or space.",
        "Friction is a force that opposes relative motion between surfaces in contact.",
        "Density is mass per unit volume of a substance.",
        "Buoyancy is the upward force exerted on an object submerged in a fluid.",
        "Thermal expansion is the tendency of matter to change volume in response to temperature.",
        "Radiation is energy emitted as electromagnetic waves or subatomic particles.",
        "An atom is the smallest unit of an element that retains its chemical properties.",
        "A molecule is two or more atoms bonded together as the smallest unit of a substance.",
        "Chemical bonding is the attraction between atoms that allows the formation of molecules.",
        "A solid has a definite shape and volume with particles closely packed together.",
        "A liquid has a definite volume but no definite shape taking the shape of its container.",
        "A gas has no definite shape or volume and expands to fill its container.",
        "Plasma is a high-temperature state of matter consisting of ionized gas.",
        "Nuclear energy is released through fission or fusion of atomic nuclei.",
        "Inertia is the resistance of an object to changes in its state of motion.",
        "Conservation of energy states that energy cannot be created or destroyed only converted.",
        "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
    ],

    "materials": [
        "A material is a substance from which objects can be made.",
        "Metal is a solid material that is typically hard shiny malleable and ductile.",
        "Steel is an alloy of iron and carbon that is strong and widely used in construction.",
        "Aluminum is a lightweight silvery metal with high strength and corrosion resistance.",
        "Concrete is a composite material made of aggregate bonded with cement and water.",
        "Wood is a hard fibrous structural tissue found in trees and other plants.",
        "Plastic is a synthetic material made from polymers that can be molded into any shape.",
        "Glass is an amorphous solid made from silica with a smooth hard transparent surface.",
        "Rubber is an elastic material that returns to its original shape after stretching.",
        "Ceramic is an inorganic non-metallic solid made by heating and cooling.",
        "A composite material is made from two or more constituent materials.",
        "Hardness is the resistance of a material to surface scratching or indentation.",
        "Tensile strength is the maximum stress a material can withstand while being stretched.",
        "Conductivity refers to the ability of a material to conduct heat or electricity.",
        "Insulation is a material that reduces the transfer of heat or electricity.",
        "Brittleness is the tendency of a material to fracture under stress.",
        "Elasticity is the ability of a material to return to its original shape after deformation.",
        "Porosity is the measure of void spaces in a material.",
        "Density is a key physical property of materials affecting strength and weight.",
        "Corrosion is the gradual deterioration of a material through chemical reaction.",
        "Alloy is a mixture of a metal with another element to improve its properties.",
        "Polymer is a large molecule made of many repeating smaller molecules called monomers.",
        "Fiber is a slender threadlike structure that can be woven into textiles.",
        "Adhesive is a substance that bonds surfaces together through surface attachment.",
        "Solubility is the ability of a substance to dissolve in a solvent.",
        "Transparency is the property of allowing light to pass through without scattering.",
        "Surface area affects how quickly a material reacts with its environment.",
        "Thermal conductivity measures how well a material transfers heat.",
        "Malleability is the ability of a material to be shaped by hammering without breaking.",
        "Melting point is the temperature at which a solid becomes liquid.",
    ],
}

# Flat list of all sentences with domain labels
ALL_SENTENCES = [
    (domain, sentence)
    for domain, sentences in DOMAINS.items()
    for sentence in sentences
]

def get_domain_text(domain: str) -> str:
    """Get all sentences for a domain as a single text block."""
    return " ".join(DOMAINS.get(domain, []))

def get_all_text() -> str:
    """Get all sentences across all domains."""
    return " ".join(s for _, s in ALL_SENTENCES)

def domain_names():
    return list(DOMAINS.keys())
