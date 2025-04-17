import { configureStore } from '@reduxjs/toolkit';
import demandesReducer from '../Features/Demande/DemandeSlice';
import motifsReducer from '../Features/Demande/MotifSlice';
import employeReducer from '../Features/Employe/EmployeSlice';
import directionReducer from '../Features/Employe/DirectionSlice';


export const store = configureStore({
  reducer: {
    direction: directionReducer,
    demandes: demandesReducer,
    motifs: motifsReducer,
    employes: employeReducer,
  },
});
